"""ZestXML vs SASRec, in one file. Paste into a Colab cell and run.

    !pip -q install scikit-learn
    # paste this file, then:
    rows = main(dataset="ml-1m", cold_frac=0.1, sasrec_epochs=200)

No repository, no imports beyond numpy / scipy / sklearn / torch. Every piece is here to
be edited: the pattern miner, the bilinear scorer, the SASRec block, the split, the metric.

WHAT THIS IS AND IS NOT
-----------------------
This is a *compact reimplementation* of ZestXML's mechanism (~200 lines), not the full
port. It reproduces: tf-idf point features, label feature bags, the exact-match direct map,
the Jaccard-style pattern miner, W confined to that pattern, and the
``alpha * bilinear + (1 - alpha) * knn`` blend. It drops the file formats, the parameter
system, the chunked kernels, propensity metrics, and the approximate shortlist -- at a few
thousand items every score matrix fits in memory, so it ranks the full catalogue directly.
It also trains against sampled negatives rather than a mined shortlist, which is the one
substantive difference in the learning signal.

So the numbers here are NOT the full port's. For reference, the full port on ml-1m
(cold_frac 0.1, 200 SASRec epochs, shortyK 500), Recall@10 / Recall@50:

    zestxml          all  6.77 / 13.81   head 10.10 / 17.30   tail 4.88 / 10.14   cold 1.34 / 10.70
    sasrec           all 19.93 / 43.05   head 30.30 / 61.37   tail 16.00 / 41.16   cold 0.00 /  0.00
    sasrec+content   all 19.59 / 43.94   head 30.00 / 61.17   tail 14.86 / 42.14   cold 0.75 /  3.51

The SASRec side of this file is the same code that produced those, so it should match. The
ZestXML side will not, and at the time of writing it had not been calibrated against them
at matched settings. Compare arms *within* one run of this file; do not mix its ZestXML row
with the table above.

THE COMPARISON
--------------
SASRec reads an ordered item-id sequence and scores an item through a learned per-item
embedding. ZestXML reads one bag of tf-idf features and scores an item through the words of
its title. SASRec sees order; ZestXML does not. An item with zero training interactions has
an untrained embedding in SASRec and a perfectly ordinary title in ZestXML. That is the
whole trade, and the cold-item column is where it shows.

Held identical across arms: the split, the catalogue, the cold set, full-catalogue ranking,
the history mask, and the metric function. ``sasrec+content`` gets the *same* tf-idf text
ZestXML gets, through a learned linear map, so the cold column is a baseline and not a
strawman.

Published SASRec ml-1m numbers (HR@10 ~ 0.82) rank against 100 sampled negatives. Krichene
& Rendle (KDD'20) showed that is not a consistent estimator of the full-catalogue metric.
Everything here ranks the full catalogue, so it is much lower and NOT comparable. Do not
put the two in one table.
"""

from __future__ import annotations

import csv
import gzip
import json
import os
import random
import urllib.request
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================================== #
# 1. datasets  --  a loader returns ({user: [item, ...] in time order},
#                                    {item: "text describing the item"})
#    Everything downstream reads only that, so a new dataset is a new loader.
# =========================================================================== #
ML1M = "https://raw.githubusercontent.com/wesm/pydata-book/master/datasets/movielens/"


def load_ml1m(root="raw/ml-1m", min_len=5):
    os.makedirs(root, exist_ok=True)
    for name in ("ratings.dat", "movies.dat"):
        path = f"{root}/{name}"
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            print(f"downloading {name} ...")
            urllib.request.urlretrieve(ML1M + name, path)

    titles = {}
    for line in open(f"{root}/movies.dat", encoding="latin-1"):
        p = line.rstrip("\n").split("::")
        if len(p) >= 3:
            titles[int(p[0])] = f"{p[1]} {p[2].replace('|', ' ')}"

    events = defaultdict(list)
    for line in open(f"{root}/ratings.dat", encoding="latin-1"):
        p = line.rstrip("\n").split("::")
        if len(p) >= 4:
            events[int(p[0])].append((int(p[3]), int(p[1])))

    seqs = {}
    for u, rows in events.items():
        rows.sort()  # by timestamp then item -- deterministic under ties
        items = [i for _, i in rows if i in titles]
        if len(items) >= min_len:
            seqs[u] = items
    return seqs, titles


AMZ = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023"


def load_amazon(root=None, category="Video_Games", min_len=5, max_users=None):
    """Amazon Reviews 2023.  NOTE: this download was never reachable from the machine
    this file was written on, so the URLs are untested (the parsing is tested).  If it
    fails, fetch the two files by hand from https://amazon-reviews-2023.github.io/ .
    """
    root = root or f"raw/amazon-{category}"
    os.makedirs(root, exist_ok=True)
    for url, name in ((f"{AMZ}/benchmark/5core/rating_only/{category}.csv.gz",
                       f"{category}.csv.gz"),
                      (f"{AMZ}/raw/meta_categories/meta_{category}.jsonl.gz",
                       f"meta_{category}.jsonl.gz")):
        path = f"{root}/{name}"
        if os.path.exists(path) and os.path.getsize(path) > 0:
            continue
        print(f"downloading {name} ...")
        try:
            urllib.request.urlretrieve(url, path)
        except Exception as exc:
            if os.path.exists(path):
                os.remove(path)
            raise SystemExit(f"could not fetch {url}\n  {exc}\n"
                             f"Download by hand into {root}/ as {name}.")

    titles = {}
    with gzip.open(f"{root}/meta_{category}.jsonl.gz", "rt", errors="replace") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key, title = row.get("parent_asin"), (row.get("title") or "").strip()
            if key and title:
                cats = row.get("categories") or []
                titles[key] = " ".join([title] + [str(c) for c in cats[:3]])

    events = defaultdict(list)
    with gzip.open(f"{root}/{category}.csv.gz", "rt", errors="replace") as f:
        for row in csv.DictReader(f):
            # parent_asin, not asin: colour and size variants are one product, and scoring
            # them apart lets a model "miss" while naming the same thing
            item = row.get("parent_asin") or row.get("asin")
            u, ts = row.get("user_id"), row.get("timestamp")
            if item in titles and u and ts:
                events[u].append((int(float(ts)), item))

    seqs = {}
    for u, rows in sorted(events.items()):
        rows.sort()
        if len(rows) >= min_len:
            seqs[u] = [i for _, i in rows]
        if max_users and len(seqs) >= max_users:
            break
    if not seqs:
        raise SystemExit(f"{root}: no user reached {min_len} interactions with a titled "
                         f"item ({len(titles)} titles, {len(events)} users read)")
    return seqs, titles


LOADERS = {"ml-1m": load_ml1m, "amazon": load_amazon}


# =========================================================================== #
# 2. split
# =========================================================================== #
class Split:
    """Leave-one-out, plus a set of items removed from training entirely.

    The cold construction is the point of the exercise, so it is worth being exact: an item
    is chosen cold, then *every* occurrence of it is deleted from every training history and
    from the validation targets. It survives only where it is somebody's final test target.
    That leaves it with exactly zero training interactions -- not few, zero.
    """

    def __init__(self, seqs, titles, cold_frac=0.1, seed=0):
        rng = random.Random(seed)
        self.titles = titles
        self.items = sorted({i for s in seqs.values() for i in s})
        self.index = {it: k for k, it in enumerate(self.items)}

        self.cold = set()
        if cold_frac > 0:
            # only items that are somebody's last interaction can be held out and still be
            # measured; a cold item nobody is tested on is noise, not signal
            pool = sorted({s[-1] for s in seqs.values() if len(s) >= 3})
            rng.shuffle(pool)
            self.cold = set(pool[:int(round(cold_frac * len(self.items)))])

        self.train, self.val, self.test, self.hist = {}, {}, {}, {}
        for u, s in seqs.items():
            if len(s) < 3:
                continue
            h = [i for i in s[:-2] if i not in self.cold]
            if len(h) < 2:
                continue
            self.train[u] = h
            if s[-2] not in self.cold:
                self.val[u] = s[-2]
            self.test[u] = s[-1]
            self.hist[u] = h + ([s[-2]] if s[-2] not in self.cold else [])

        self.count = np.zeros(len(self.items), dtype=np.int64)
        for h in self.train.values():
            for i in h:
                self.count[self.index[i]] += 1
        for v in self.val.values():
            self.count[self.index[v]] += 1

        self.users = sorted(self.test)
        self.targets = np.array([self.index[self.test[u]] for u in self.users])
        self.test_hist = [[self.index[i] for i in self.hist[u]] for u in self.users]

    @property
    def n_items(self):
        return len(self.items)

    def groups(self, head_frac=0.2):
        warm = np.nonzero(self.count > 0)[0]
        order = warm[np.argsort(-self.count[warm])]
        head = np.zeros(self.n_items, dtype=bool)
        head[order[:int(round(head_frac * len(order)))]] = True
        return {"head": head, "tail": (self.count > 0) & ~head, "cold": self.count == 0}


# =========================================================================== #
# 3. one metric function, used by every arm
# =========================================================================== #
def evaluate(scores, split, groups, ks=(10, 50)):
    """``scores`` dense (n_test_users, n_items).  Items already in the user's history are
    masked: ZestXML has no notion of a sequence and would otherwise re-recommend what the
    user just watched, which is not a prediction."""
    scores = torch.as_tensor(scores).clone().float().cpu()
    for r, h in enumerate(split.test_hist):
        if h:
            scores[r, torch.as_tensor(h)] = -np.inf

    top = torch.topk(scores, max(ks), dim=1).indices.numpy()
    tgt = split.targets
    hit = top == tgt[:, None]
    rank = np.where(hit.any(1), hit.argmax(1), max(ks) + 1)

    out = {}
    for name, mask in [("all", np.ones(len(tgt), bool))] + [(g, m[tgt]) for g, m in groups.items()]:
        if mask.sum() == 0:
            continue
        r = rank[mask]
        row = {"n": int(mask.sum())}
        for k in ks:
            row[f"Recall@{k}"] = 100.0 * float((r < k).mean())
        row["NDCG@10"] = 100.0 * float(
            np.where(r < 10, 1.0 / np.log2(r.astype(float) + 2), 0.0).mean())
        row["MRR"] = 100.0 * float(np.where(r <= max(ks), 1.0 / (r + 1.0), 0.0).mean())
        out[name] = row
    out["all"]["coverage@10"] = 100.0 * len(np.unique(top[:, :10])) / scores.shape[1]
    return out


def print_table(rows):
    cols = ["Recall@10", "Recall@50", "NDCG@10", "MRR"]
    print("%-18s %-6s %6s " % ("model", "group", "n") + " ".join("%9s" % c for c in cols))
    for model, groups in rows.items():
        for g, m in groups.items():
            print("%-18s %-6s %6d " % (model, g, m["n"])
                  + " ".join("%9.2f" % m[c] for c in cols))
    print("\ncoverage@10: " + ", ".join(f"{k} {v['all']['coverage@10']:.1f}%"
                                        for k, v in rows.items()))


# =========================================================================== #
# 4. ZestXML
# =========================================================================== #
# score(x, y) = alpha * sigma(x^T W y) + (1 - alpha) * knn(x, y)
#
#   * a label is a *bag of text features*, not an id.  Item "Toy Story (1995) Animation"
#     emits one feature per token plus one unique identity feature.  That is the entire
#     zero-shot mechanism: a cold item shares "animation" with warm ones.
#   * W is confined to a mined sparsity pattern.  The free parameters are one scalar per
#     pattern entry -- not a |Xf| x |Yf| matrix.
#   * knn is the *direct map*: point feature "animation" linked to label feature
#     "1_animation" by string equality.  It is untrained, so it works from step zero.
# --------------------------------------------------------------------------- #
def label_features(names):
    """Yf: one shared token feature per word, plus a unique identity feature per label.

    The identity feature is what lets a well-observed item be memorised; the token features
    are what let an unobserved one be scored at all.  Drop the identity features and the
    head column collapses; drop the token features and the cold column does.
    """
    vocab, rows, cols = {}, [], []
    for i, name in enumerate(names):
        feats = [f"1_{t}" for t in name.lower().split() if t.isalnum()]
        feats.append(f"__label__{i}")
        for f in feats:
            rows.append(i)
            cols.append(vocab.setdefault(f, len(vocab)))
    Y = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                      shape=(len(names), len(vocab)))
    return Y, [f for f, _ in sorted(vocab.items(), key=lambda kv: kv[1])]


def direct_map(Xf, Yf, weight):
    """Link label feature "1_tok" to point feature "tok" by exact string equality.

    The label-feature name carries a prefix up to the first underscore; whatever follows is
    the lookup key.  Identity features never match anything, by construction.
    """
    xi = {name: i for i, name in enumerate(Xf)}
    rows, cols = [], []
    for j, name in enumerate(Yf):
        key = name[name.find("_") + 1:]
        if key in xi:
            rows.append(xi[key])
            cols.append(j)
    return sp.csr_matrix((np.full(len(rows), weight, np.float32), (rows, cols)),
                         shape=(len(Xf), len(Yf)))


def _topk_rows(M, freq_r, freq_c, af_r, af_c, alpha, k):
    """Jaccard-ish score, then top-k per row.

        score = cooc / (alpha * (freq_r + freq_c - cooc) + (1 - alpha) * af_r * af_c)

    The first denominator term is the Jaccard union; the second is a product of marginals
    that damps pairs which co-occur only because both are frequent.  ``alpha`` mixes them.
    """
    M = M.tocsr()
    rows, cols, vals = [], [], []
    for r in range(M.shape[0]):
        lo, hi = M.indptr[r], M.indptr[r + 1]
        if lo == hi:
            continue
        c, v = M.indices[lo:hi], M.data[lo:hi]
        denom = alpha * (freq_r[r] + freq_c[c] - v) + (1 - alpha) * af_r[r] * af_c[c]
        s = np.where(v > 0, v / np.maximum(denom, 1e-12), 0.0)
        if len(s) > k:
            keep = np.argpartition(-s, k)[:k]
            c, s = c[keep], s[keep]
        good = s > 0
        rows.append(np.full(good.sum(), r))
        cols.append(c[good])
        vals.append(s[good])
    if not rows:
        return sp.csr_matrix((M.shape[0], M.shape[1]), dtype=np.float32)
    return sp.csr_matrix((np.concatenate(vals).astype(np.float32),
                          (np.concatenate(rows), np.concatenate(cols))), shape=M.shape)


def mine_pattern(X, Y, XY, Xf, Yf, bs_count=20, bs_alpha=0.02, direct_wt=0.8):
    """Which (point feature, label feature) pairs W is allowed to be non-zero on."""
    Xb, Yb, XYb = (m.copy() for m in (X, Y, XY))
    for m in (Xb, Yb, XYb):
        m.data[:] = 1.0
    X_Yf = (XYb @ Yb).tocsr()          # label features seen per point
    Y_Xf = (XYb.T @ Xb).tocsr()        # point features seen per label

    Yf_freq = np.asarray(X_Yf.sum(0)).ravel()
    Xf_freq = np.asarray(Y_Xf.sum(0)).ravel()
    Xf_df = np.asarray(Xb.sum(0)).ravel()
    Yf_df = np.asarray(Yb.sum(0)).ravel()

    Xf_Yf = _topk_rows(Xb.T @ X_Yf, Xf_freq, Yf_freq, Xf_df, Yf_df, bs_alpha, bs_count)
    Yf_Xf = _topk_rows(Yb.T @ Y_Xf, Yf_freq, Xf_freq, Yf_df, Xf_df, bs_alpha, bs_count)

    direct = direct_map(Xf, Yf, direct_wt)
    pattern = (Xf_Yf + Yf_Xf.T + direct).tocoo()
    pattern.sum_duplicates()
    return pattern, direct


class ZestXML:
    """W over the mined pattern, trained with a squared hinge against sampled negatives.

    At these catalogue sizes every label is scored for every point, so there is no
    shortlist: the "candidate generation" stage of the full system is replaced by ranking
    everything, which strictly removes a source of error rather than adding one.  If you
    push this past ~50k items, that is the first thing to put back.
    """

    def __init__(self, pattern, direct, n_xf, n_yf, alpha=0.9):
        self.idx = torch.as_tensor(np.stack([pattern.row, pattern.col]), device=DEVICE)
        self.shape = (n_xf, n_yf)
        self.alpha = alpha
        self.w = torch.zeros(len(pattern.data), device=DEVICE, requires_grad=True)
        self.b = torch.zeros(1, device=DEVICE, requires_grad=True)
        d = direct.tocoo()
        scale = max(float(np.abs(d.data).max()) if d.nnz else 1.0, 1e-12)
        self.direct_t = torch.sparse_coo_tensor(
            np.stack([d.col, d.row]), d.data / scale, self.shape[::-1],
            device=DEVICE, dtype=torch.float32).coalesce()

    def _Wt(self):
        """W transposed, built by swapping the index rows rather than calling .t().

        torch.sparse.mm(sparse, dense) carries the gradient back to the sparse values, and
        that is the only reason this never materialises the |Xf| x |Yf| matrix. Keep the
        second argument DENSE: sparse @ sparse returns a sparse result, and adding a dense
        bias to it raises "add(sparse, dense) is not supported".
        """
        return torch.sparse_coo_tensor(self.idx.flip(0), self.w,
                                       self.shape[::-1]).coalesce()

    def _margins(self, Xb, Y, Wt):
        """(batch, n_items) margins.  ``Xb`` dense (batch, n_xf), ``Y`` sparse."""
        proj = torch.sparse.mm(Wt, Xb.t())             # (n_yf, batch) dense
        return torch.sparse.mm(Y, proj).t()            # (batch, n_items) dense

    def fit(self, X, Y, XY, epochs=20, lr=0.2, batch=256, cost=5.0, negs=64, seed=0, log=print):
        g = torch.Generator().manual_seed(seed)
        Xt = _to_torch(X)
        Yt = _to_torch(Y)
        truth = XY.tocsr()
        opt = torch.optim.Adam([self.w, self.b], lr=lr)
        n = X.shape[0]
        for ep in range(epochs):
            for pg in opt.param_groups:
                pg["lr"] = lr * (1 - ep / max(1, epochs))
            order = torch.randperm(n, generator=g).numpy()
            total = 0.0
            for lo in range(0, n, batch):
                rows = order[lo:lo + batch]
                Xb = _dense(X[rows])
                m = self._margins(Xb, Yt, self._Wt()) + self.b

                # positives from the truth, plus a sample of negatives -- scoring every
                # item every step is affordable here but the gradient is dominated by
                # easy negatives, and sampling is what the original does anyway
                tgt = torch.zeros_like(m)
                pr, pc = [], []
                for r, i in enumerate(rows):
                    lbl = truth.indices[truth.indptr[i]:truth.indptr[i + 1]]
                    pr += [r] * len(lbl)
                    pc += list(lbl)
                tgt[pr, pc] = 1.0
                pick = torch.randint(m.shape[1], (len(rows), negs), generator=g).to(m.device)
                sel = torch.zeros_like(m, dtype=torch.bool)
                sel[torch.arange(len(rows), device=m.device)[:, None], pick] = True
                sel[pr, pc] = True

                y = 2 * tgt - 1
                hinge = torch.clamp(1 - y * m, min=0) ** 2
                wt = torch.where(tgt > 0, cost * 10.0, cost)
                loss = (hinge * wt * sel).sum() / sel.sum().clamp_min(1)
                loss = loss + 0.5 * (self.w.pow(2).sum() + self.b.pow(2).sum()) / n
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total += float(loss.detach())
            if log and (ep + 1) % 5 == 0:
                log(f"  zestxml epoch {ep + 1}/{epochs} loss {total / max(1, n // batch):.4f}")
        return self

    @torch.no_grad()
    def scores(self, X, Y, batch=256):
        Yt, Wt = _to_torch(Y), self._Wt()
        out = torch.zeros(X.shape[0], Y.shape[0])
        for lo in range(0, X.shape[0], batch):
            Xb = _dense(X[lo:lo + batch])
            m = self._margins(Xb, Yt, Wt) + self.b
            bil = torch.exp(-torch.clamp(1 - m, min=0) ** 2)   # margin -> positive score
            knn = self._margins(Xb, Yt, self.direct_t)
            out[lo:lo + batch] = (self.alpha * bil + (1 - self.alpha) * knn).cpu()
        return out


def _dense(m):
    """A batch of points as a dense (batch, n_xf) tensor.

    Dense on purpose -- see :meth:`ZestXML._Wt`. At 256 x ~30k features this is ~30 MB; if
    you run this on a vocabulary ten times larger, drop ``batch`` rather than going sparse.
    """
    return torch.as_tensor(np.asarray(m.todense(), dtype=np.float32), device=DEVICE)


def _to_torch(m):
    m = m.tocoo()
    return torch.sparse_coo_tensor(np.stack([m.row, m.col]), m.data.astype(np.float32),
                                   m.shape, device=DEVICE, dtype=torch.float32).coalesce()


def run_zestxml(split, ctx=20, windows=8, seed=0, epochs=20, alpha=0.9,
                bs_count=20, bs_alpha=0.02, direct_wt=0.8, min_df=2, log=print):
    """One point is a history prefix; its label is the next item.

    ``ctx`` bounds how many recent items form the text -- a 200-item history averaged into
    one tf-idf vector is mostly noise, and this recency cut is the *only* sequence
    information a bag-of-words model retains.
    """
    rng = random.Random(seed)
    names = _unique_names(split)
    text = {i: split.titles.get(i, "") for i in split.items}

    def doc(prefix):
        return " ".join(text[i] for i in prefix[-ctx:])

    trn_x, trn_y = [], []
    for u, h in split.train.items():
        full = h + ([split.val[u]] if u in split.val else [])
        pos = list(range(1, len(full)))
        if len(pos) > windows:
            pos = sorted(rng.sample(pos, windows))
        for k in pos:
            trn_x.append(doc(full[:k]))
            trn_y.append(split.index[full[k]])

    tst_x = [doc(split.hist[u]) for u in split.users]

    vec = TfidfVectorizer(lowercase=True, token_pattern=r"[a-z][a-z0-9]+",
                          ngram_range=(1, 2), min_df=min_df, sublinear_tf=True,
                          stop_words="english")
    Xtr = _rownorm(vec.fit_transform(trn_x))
    Xte = _rownorm(vec.transform(tst_x))
    Xf = vec.get_feature_names_out().tolist()
    Y, Yf = label_features(names)
    Y = _rownorm(Y)

    XY = sp.csr_matrix((np.ones(len(trn_y), np.float32),
                        (np.arange(len(trn_y)), np.array(trn_y))),
                       shape=(len(trn_y), split.n_items))

    # Cold items must be invisible to the miner: their features stay (that is what makes
    # them scorable) but they contribute no co-occurrence counts, because they have no
    # training interaction to count.  XY already has no column for them.
    log(f"  {Xtr.shape[0]} points, {len(Xf)} point features, {len(Yf)} label features")
    pattern, direct = mine_pattern(Xtr, Y, XY, Xf, Yf, bs_count, bs_alpha, direct_wt)
    log(f"  pattern nnz {pattern.nnz} ({len(direct.data)} from the direct map)")

    model = ZestXML(pattern, direct, len(Xf), len(Yf), alpha=alpha)
    model.fit(Xtr, Y, XY, epochs=epochs, seed=seed, log=log)
    return model.scores(Xte, Y)


def _unique_names(split):
    """Label names, forced unique -- two items sharing a title would otherwise merge."""
    names, seen, clash = [], {}, 0
    for it in split.items:
        base = split.titles.get(it, f"item {it}")
        if base in seen:
            clash += 1
            base = f"{base} v{seen[base] + 1}"
        seen[base] = seen.get(base, 0) + 1
        names.append(base)
    if clash:
        print(f"  note: {clash} duplicate titles disambiguated")
    return names


def _rownorm(M):
    M = sp.csr_matrix(M, dtype=np.float32)
    n = np.sqrt(np.asarray(M.multiply(M).sum(1)).ravel())
    return sp.diags(1.0 / np.maximum(n, 1e-12)) @ M


# =========================================================================== #
# 5. SASRec
# =========================================================================== #
class SASRec(nn.Module):
    """Kang & McAuley (ICDM'18), compactly.

    ``content`` optionally adds a fixed per-item vector through a learned linear map, so a
    cold item has a non-random representation.  Without it the cold column measures "an
    untrained embedding table", which is true about SASRec but a weak baseline.
    """

    def __init__(self, n_items, d=50, maxlen=200, blocks=2, heads=1, dropout=0.2,
                 content=None):
        super().__init__()
        self.item = nn.Embedding(n_items + 1, d, padding_idx=0)   # 0 is the pad slot
        self.pos = nn.Embedding(maxlen, d)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(d, heads, d * 4, dropout, batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, blocks, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.proj = None
        if content is not None:
            self.register_buffer("content", content)
            self.proj = nn.Linear(content.shape[1], d, bias=False)
        nn.init.normal_(self.item.weight, std=0.02)
        with torch.no_grad():
            self.item.weight[0].zero_()

    def table(self):
        v = self.item.weight
        if self.proj is not None:
            extra = torch.zeros_like(v)
            extra[1:] = self.proj(self.content)
            v = v + extra
        return v

    def forward(self, seq):
        n = seq.shape[1]
        t = self.table()
        x = t[seq] * (t.shape[1] ** 0.5)
        x = self.drop(x + self.pos(torch.arange(n, device=seq.device))[None])
        x = x * (seq > 0).unsqueeze(-1)
        causal = torch.triu(torch.ones(n, n, dtype=torch.bool, device=seq.device), 1)
        return self.norm(torch.nan_to_num(
            self.encoder(x, mask=causal, src_key_padding_mask=(seq == 0))))


def run_sasrec(split, content=None, d=50, maxlen=200, epochs=200, lr=1e-3, batch=128,
               seed=0, log=print):
    torch.manual_seed(seed)
    ct = None if content is None else torch.as_tensor(content, dtype=torch.float32,
                                                      device=DEVICE)
    model = SASRec(split.n_items, d=d, maxlen=maxlen, content=ct).to(DEVICE)

    users = sorted(split.train)
    seq = np.zeros((len(users), maxlen), np.int64)
    pos = np.zeros((len(users), maxlen), np.int64)
    for r, u in enumerate(users):
        h = [split.index[i] + 1 for i in split.train[u]]        # 1-based; 0 is pad
        if u in split.val:
            h.append(split.index[split.val[u]] + 1)
        h = h[-(maxlen + 1):]
        seq[r, maxlen - len(h) + 1:] = h[:-1]
        pos[r, maxlen - len(h) + 1:] = h[1:]
    seq_t, pos_t = torch.as_tensor(seq, device=DEVICE), torch.as_tensor(pos, device=DEVICE)

    # negatives come from items with a training interaction: sampling a cold item as a
    # negative trains the model to push down exactly what the cold column then asks it to
    # rank up, which would rig the comparison
    warm = torch.as_tensor(np.nonzero(split.count > 0)[0] + 1, device=DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98))
    g = torch.Generator().manual_seed(seed)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    for ep in range(epochs):
        model.train()
        order = torch.randperm(len(users), generator=g).to(DEVICE)
        tot, seen = 0.0, 0
        for lo in range(0, len(users), batch):
            b = order[lo:lo + batch]
            state, t = model(seq_t[b]), model.table()
            p = pos_t[b]
            neg = warm[torch.randint(warm.numel(), p.shape, device=DEVICE)]
            mask = p > 0
            lp = (state * t[p]).sum(-1)
            ln = (state * t[neg]).sum(-1)
            loss = bce(lp, torch.ones_like(lp)) + bce(ln, torch.zeros_like(ln))
            loss = (loss * mask).sum() / mask.sum().clamp_min(1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * int(mask.sum())
            seen += int(mask.sum())
        if log and (ep + 1) % 20 == 0:
            log(f"  sasrec epoch {ep + 1}/{epochs} loss {tot / max(1, seen):.4f}")

    model.eval()
    seq = np.zeros((len(split.users), maxlen), np.int64)
    for r, u in enumerate(split.users):
        h = [split.index[i] + 1 for i in split.hist[u]][-maxlen:]
        seq[r, maxlen - len(h):] = h
    out = torch.zeros(len(split.users), split.n_items)
    with torch.no_grad():
        t = model.table()[1:]
        for lo in range(0, len(split.users), 256):
            b = torch.as_tensor(seq[lo:lo + 256], device=DEVICE)
            out[lo:lo + 256] = (model(b)[:, -1] @ t.T).cpu()
    return out


def content_vectors(split, dim=128, seed=0):
    """A fixed content vector per item, from the *same* text ZestXML reads.

    Deliberately not a pretrained sentence encoder: the point is to give SASRec the same
    information, so a difference between the two is about the model and not about who got
    the better features.
    """
    texts = [split.titles.get(i, "") for i in split.items]
    tf = TfidfVectorizer(lowercase=True, token_pattern=r"[a-z][a-z0-9]+", ngram_range=(1, 2),
                         min_df=1, sublinear_tf=True, stop_words="english").fit_transform(texts)
    v = TruncatedSVD(min(dim, tf.shape[1] - 1), random_state=seed).fit_transform(tf)
    v = v.astype(np.float32)
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-8)


# =========================================================================== #
# 6. runner
# =========================================================================== #
def main(dataset="ml-1m", cold_frac=0.1, sasrec_epochs=200, zest_epochs=20, ctx=20,
         windows=8, seed=0, arms=("zestxml", "sasrec", "sasrec+content"), **loader_kw):
    seqs, titles = LOADERS[dataset](**loader_kw)
    split = Split(seqs, titles, cold_frac=cold_frac, seed=seed)
    groups = split.groups()
    print(f"{len(split.train)} users, {split.n_items} items, {len(split.cold)} cold "
          f"({100.0 * len(split.cold) / split.n_items:.1f}%), {len(split.users)} test "
          f"points, {int(groups['cold'][split.targets].sum())} of them cold")

    rows = {}
    if "zestxml" in arms:
        print("\n=== zestxml")
        rows["zestxml"] = evaluate(
            run_zestxml(split, ctx=ctx, windows=windows, seed=seed, epochs=zest_epochs),
            split, groups)
    content = content_vectors(split, seed=seed)
    for name, ct in (("sasrec", None), ("sasrec+content", content)):
        if name in arms:
            print(f"\n=== {name}")
            rows[name] = evaluate(
                run_sasrec(split, content=ct, epochs=sasrec_epochs, seed=seed),
                split, groups)

    print("\n" + "=" * 72)
    print_table(rows)
    return rows


if __name__ == "__main__":
    main()
