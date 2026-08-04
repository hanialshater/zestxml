"""ZestXML vs SASRec, in one file. Paste into a Colab cell and run.

    !pip -q install scikit-learn
    # paste this file, then:
    rows = main(dataset="ml-1m", cold_frac=0.1, horizon=5, sasrec_epochs=200)

No repository, no imports beyond numpy / scipy / sklearn / torch. Every piece is here to
be edited: the pattern miner, the bilinear scorer, the SASRec block, the split, the metric.

WHAT THIS IS AND IS NOT
-----------------------
A compact reimplementation of ZestXML's mechanism (~250 lines), not the full port. It
reproduces the parts that decide the numbers: tf-idf point features, label feature bags,
the exact-match direct map, the two-direction Jaccard pattern miner with de-duplication,
W confined to that pattern, features of a zero-interaction label blanked for training and
restored at prediction, training on mined shortlist pairs with the weighted squared hinge,
prediction restricted to the shortlist, and the ``alpha * bilinear + (1 - alpha) * knn``
blend. It drops the file formats, the parameter system and the chunked kernels: margins are
computed densely per batch and masked to the shortlist, which is exactly equivalent below
roughly 100k labels and much shorter.

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

THE TASK IS XMC, NOT NEXT-ITEM
------------------------------
A point is a user and its labels are the SET of items they touch in the next ``horizon``
steps. Metrics are the extreme-classification suite -- P@k, nDCG@k, and propensity-scored
PSP@k -- with the same definitions as ``zestxml/eval.py``, so they line up with the rest of
that repository. Read PSP@k, not just P@k: it is the column that rewards retrieving *rare*
labels, and a popularity-follower scores well on P@k while doing nothing useful.

Consequence for SASRec: it still *trains* on next-item, because that is what the
architecture is for and crippling it would not make the comparison fairer. Only the
evaluation is shared.

Published SASRec ml-1m numbers (HR@10 ~ 0.82) are leave-one-out against 100 sampled
negatives and have nothing to do with anything here -- different task, different candidate
set. Do not put the two in one table.
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
# 2. split  --  XMC framing: one point per user, a SET of future items as labels
# =========================================================================== #
class Split:
    """Time-ordered split with a future *window* as the target, plus a cold item set.

    This is an extreme-multi-label problem, not leave-one-out next-item. A point is a user;
    its labels are every item they interact with in the next ``horizon`` steps. That is what
    the metrics below assume, and it is the framing the comparison is actually about --
    "which items does this customer want", not "which single item is literally next".

    Layout per user, in time order::

        [ ....... train history ....... | val window | test window ]
                                          horizon      horizon

    The cold construction is the point of the exercise, so it is worth being exact: an item
    is chosen cold, then *every* occurrence of it is deleted from the training history and
    the validation window. It survives only inside test windows. That leaves it with
    exactly zero training interactions -- not few, zero.
    """

    def __init__(self, seqs, titles, cold_frac=0.1, horizon=5, seed=0):
        rng = random.Random(seed)
        self.titles, self.horizon = titles, horizon
        self.items = sorted({i for s in seqs.values() for i in s})
        self.index = {it: k for k, it in enumerate(self.items)}
        H = horizon

        self.cold = set()
        if cold_frac > 0:
            # only items that land in somebody's test window can be held out and still be
            # measured; a cold item nobody is tested on is noise, not signal
            pool = sorted({i for s in seqs.values() if len(s) >= 2 * H + 2 for i in s[-H:]})
            rng.shuffle(pool)
            self.cold = set(pool[:int(round(cold_frac * len(self.items)))])

        self.train, self.val, self.test, self.hist = {}, {}, {}, {}
        for u, s in seqs.items():
            if len(s) < 2 * H + 2:
                continue
            head, vwin, twin = s[:-2 * H], s[-2 * H:-H], s[-H:]
            head = [i for i in head if i not in self.cold]
            vwin = [i for i in vwin if i not in self.cold]
            if len(head) < 2 or not twin:
                continue
            self.train[u] = head
            self.val[u] = vwin
            self.test[u] = sorted(set(twin))
            self.hist[u] = head + vwin          # everything visible at test time

        self.count = np.zeros(len(self.items), dtype=np.int64)
        for h in list(self.train.values()) + list(self.val.values()):
            for i in h:
                self.count[self.index[i]] += 1

        self.users = sorted(self.test)
        rows = [r for r, u in enumerate(self.users) for _ in self.test[u]]
        cols = [self.index[i] for u in self.users for i in self.test[u]]
        self.truth = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                                   shape=(len(self.users), self.n_items))
        self.test_hist = [[self.index[i] for i in self.hist[u]] for u in self.users]

    @property
    def n_items(self):
        return len(self.items)

    def groups(self, head_frac=0.2):
        """Label masks. Metrics are reported per group by masking *labels*, the way the
        reference splits seen from unseen -- not by bucketing points."""
        warm = np.nonzero(self.count > 0)[0]
        order = warm[np.argsort(-self.count[warm])]
        head = np.zeros(self.n_items, dtype=bool)
        head[order[:int(round(head_frac * len(order)))]] = True
        return {"head": head, "tail": (self.count > 0) & ~head, "cold": self.count == 0}


# =========================================================================== #
# 3. XMC evaluation  --  P@k, nDCG@k, propensity-scored PSP@k
#    Same definitions as zestxml/eval.py, so the numbers line up with the rest
#    of that repository rather than being a private convention.
# =========================================================================== #
KS = (1, 3, 5)
COLUMNS = ["P@1", "P@3", "P@5", "nDCG@5", "PSP@1", "PSP@3", "PSP@5"]


def inv_propensity(count, n_points, A=0.55, B=1.5):
    """Jain et al.'s propensity model. PSP@k rewards retrieving *rare* labels, which is
    where a text-scored model earns its keep and a popularity-follower does not -- so it
    is the column to read if P@k looks like a rout."""
    C = (np.log(max(n_points, 2)) - 1) * (B + 1) ** A
    return 1.0 + C * np.exp(-A * np.log(count + B))


def _metrics(scores, truth, inv_prop, ks=KS):
    ks = [k for k in ks if k <= scores.shape[1]]
    keep = truth.sum(1) > 0          # points with nothing to retrieve carry no information
    scores, truth = scores[keep], truth[keep]
    if scores.shape[0] == 0:
        return None
    out = {"points": int(keep.sum())}
    top = torch.topk(scores, max(ks), dim=1).indices
    hits = torch.gather(truth, 1, top)
    ip = torch.as_tensor(inv_prop, dtype=torch.float32)
    gain = torch.gather(ip[None, :].expand_as(truth), 1, top) * hits

    for k in ks:
        out[f"P@{k}"] = 100.0 * hits[:, :k].sum(1).div(k).mean().item()
        disc = 1.0 / torch.log2(torch.arange(k, dtype=torch.float) + 2)
        dcg = (hits[:, :k] * disc[None, :]).sum(1)
        n_true = truth.sum(1).clamp(max=k).long()
        ideal = torch.cat([torch.zeros(1), disc.cumsum(0)])[n_true]
        out[f"nDCG@{k}"] = 100.0 * (dcg / ideal).mean().item()
        # PSP@k: achieved propensity-weighted gain over the best achievable one
        best = torch.sort(truth * ip[None, :], dim=1, descending=True).values[:, :k]
        out[f"PSP@{k}"] = 100.0 * gain[:, :k].sum().item() / max(best.sum().item(), 1e-9)
    return out


def evaluate(scores, split, groups):
    """``scores`` dense (n_users, n_items). One row per label group.

    Items already in the user's history are masked out: ZestXML has no notion of a
    sequence and would otherwise re-recommend what the user just consumed, which is not a
    prediction. A group row masks every label outside the group to -inf on the score side
    and to zero on the truth side, exactly as the reference does for seen/unseen.
    """
    scores = torch.as_tensor(scores).clone().float().cpu()
    for r, h in enumerate(split.test_hist):
        if h:
            scores[r, torch.as_tensor(h)] = -np.inf
    truth = torch.as_tensor(np.asarray(split.truth.todense()), dtype=torch.float32)
    inv_prop = inv_propensity(split.count.astype(np.float64), len(split.train))

    rows = {"all": _metrics(scores, truth, inv_prop)}
    for name, mask in groups.items():
        m = torch.as_tensor(mask)
        s = scores.clone()
        s[:, ~m] = -np.inf
        r = _metrics(s, truth * m[None, :].float(), inv_prop)
        if r is not None:
            rows[name] = r
    return rows


def print_table(rows):
    print("%-18s %-6s %7s " % ("model", "group", "points")
          + " ".join("%7s" % c for c in COLUMNS))
    for model, groups in rows.items():
        for g, m in groups.items():
            print("%-18s %-6s %7d " % (model, g, m["points"])
                  + " ".join("%7.2f" % m[c] for c in COLUMNS))


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
    """Which ``(point feature, label feature)`` pairs W may be non-zero on.

    Mined in both directions and then de-duplicated, exactly as the reference does:
    ``Xf_Yf`` keeps the top ``bs_count`` label features per point feature, ``Yf_Xf`` the
    top ``bs_count`` point features per label feature, and an entry present in both is
    kept once. Mining only one direction leaves rare features on the losing side with no
    entries at all, which is the reason the reference does both.
    """
    Xb, Yb, XYb = (m.copy() for m in (X, Y, XY))
    for m in (Xb, Yb, XYb):
        m.data[:] = 1.0
    X_Yf = (XYb @ Yb).tocsr()          # label features seen per point
    Y_Xf = (XYb.T @ Xb).tocsr()        # point features seen per label

    # two kinds of frequency: over training (point, label) pairs, and plain document
    # frequency. The scoring formula uses one in each denominator term.
    Yf_freq = np.asarray(X_Yf.sum(0)).ravel()
    Xf_freq = np.asarray(Y_Xf.sum(0)).ravel()
    Xf_df = np.asarray(Xb.sum(0)).ravel()
    Yf_df = np.asarray(Yb.sum(0)).ravel()

    Xf_Yf = _topk_rows(Xb.T @ X_Yf, Xf_freq, Yf_freq, Xf_df, Yf_df, bs_alpha, bs_count)
    Yf_Xf = _topk_rows(Yb.T @ Y_Xf, Yf_freq, Xf_freq, Yf_df, Xf_df, bs_alpha, bs_count)

    Xf_Yf = (Xf_Yf + direct_map(Xf, Yf, direct_wt)).tocsr()
    # drop (yf, xf) entries whose transpose is already in Xf_Yf, so the two supports are
    # disjoint and every (xf, yf) gets exactly one weight
    T = Yf_Xf.T.tocsr()
    dup = T.multiply(Xf_Yf != 0)
    Yf_Xf_T = (T - dup).tocsr()
    Yf_Xf_T.eliminate_zeros()

    pattern = (Xf_Yf + Yf_Xf_T).tocoo()          # the support W lives on
    return pattern, direct_map(Xf, Yf, direct_wt).tocsr()


def shortlist(X, Y, pattern, K, batch=512):
    """Top-``K`` labels per point under ``X @ pattern @ Y^T`` -- the candidate set.

    This doubles as the untrained score the whole model starts from: it is the quantity
    the reference's ``get_shortlist`` computes, and a label outside a point's shortlist is
    never scored and therefore never retrieved. Shortlist recall is consequently a hard
    ceiling on every metric, which is why it is printed.
    """
    Pt = _to_torch(pattern.tocsr().T.tocsr())    # (n_yf, n_xf)
    Yt = _to_torch(Y)
    rows, cols = [], []
    for lo in range(0, X.shape[0], batch):
        Xb = _dense(X[lo:lo + batch])
        sc = torch.sparse.mm(Yt, torch.sparse.mm(Pt, Xb.t())).t()   # (b, n_labels)
        k = min(K, sc.shape[1])
        val, idx = torch.topk(sc, k, dim=1)
        keep = val > 0                            # zero means no shared label feature
        r = torch.arange(sc.shape[0], device=sc.device)[:, None].expand_as(idx)
        rows.append((r[keep] + lo).cpu().numpy())
        cols.append(idx[keep].cpu().numpy())
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    return sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                         shape=(X.shape[0], Y.shape[0]))


class ZestXML:
    """W over the mined pattern, trained on shortlist pairs with a squared hinge.

    Faithful to the reference in the three places that decide the numbers:

    * **only shortlist pairs contribute to the loss.** Training against every item instead
      changes the negative distribution completely -- and on this task it also samples
      *cold* items as negatives, teaching the model to push down exactly what the cold
      column then asks it to rank up.
    * **the objective** is ``0.5||w||^2 + sum_i C_i * max(0, 1 - y_i * margin_i)^2`` with
      ``C_i = cost`` (times ``pos_wt`` for positives), minimised with Adam and a linearly
      decayed step, over mini-batches of points.
    * **prediction is restricted to the shortlist.** A label outside it scores zero.

    The margins are computed densely per batch and then masked to the shortlist, which is
    exactly equivalent at these catalogue sizes and much shorter than the chunked kernel
    the reference needs above ~100k labels.
    """

    def __init__(self, pattern, direct, n_xf, n_yf, alpha=0.9):
        idx = np.stack([pattern.row, pattern.col])
        self.idx = torch.as_tensor(idx, device=DEVICE)
        self.shape = (n_xf, n_yf)
        self.alpha = alpha
        self.w = torch.zeros(pattern.nnz, device=DEVICE, requires_grad=True)
        self.b = torch.zeros(1, device=DEVICE, requires_grad=True)
        d = direct.tocoo()
        # the reference sets every direct weight to 1; dividing by the max reproduces that
        # for an exact map and keeps any fuzzy link proportional to its similarity
        scale = max(float(np.abs(d.data).max()) if d.nnz else 1.0, 1e-12)
        self.direct_t = torch.sparse_coo_tensor(
            np.stack([d.col, d.row]), d.data / scale, self.shape[::-1],
            device=DEVICE, dtype=torch.float32).coalesce()

    def _Wt(self):
        """W transposed, built by swapping index rows rather than calling ``.t()``.

        ``torch.sparse.mm(sparse, dense)`` carries the gradient back to the sparse values,
        which is the only reason this never materialises the |Xf| x |Yf| matrix. Keep the
        second argument DENSE: sparse @ sparse returns sparse, and adding a dense bias to
        that raises "add(sparse, dense) is not supported".
        """
        return torch.sparse_coo_tensor(self.idx.flip(0), self.w,
                                       self.shape[::-1]).coalesce()

    def _margins(self, Xb, Yt, Wt):
        return torch.sparse.mm(Yt, torch.sparse.mm(Wt, Xb.t())).t()

    def fit(self, X, Y, XY, cand, epochs=20, lr=0.2, batch=256, cost=5.0, pos_wt=1.0,
            seed=0, log=print):
        g = torch.Generator().manual_seed(seed)
        Yt = _to_torch(Y)
        truth, cand = XY.tocsr(), cand.tocsr()
        n_pairs = max(1, cand.nnz)
        opt = torch.optim.Adam([self.w, self.b], lr=lr)
        n = X.shape[0]
        for ep in range(epochs):
            for pg in opt.param_groups:      # linear decay, so late epochs settle
                pg["lr"] = lr * (1 - ep / max(1, epochs))
            order = torch.randperm(n, generator=g).numpy()
            total = 0.0
            for lo in range(0, n, batch):
                rows = order[lo:lo + batch]
                sel = torch.as_tensor(np.asarray(cand[rows].todense()) != 0, device=DEVICE)
                if not sel.any():
                    continue
                tgt = torch.as_tensor(np.asarray(truth[rows].todense()) > 0, device=DEVICE)
                m = self._margins(_dense(X[rows]), Yt, self._Wt()) + self.b

                y = torch.where(tgt, 1.0, -1.0)
                hinge = torch.clamp(1 - y * m, min=0) ** 2
                C = torch.where(tgt, cost * pos_wt, cost)
                data = (hinge * C * sel).sum()
                reg = 0.5 * (self.w.pow(2).sum() + self.b.pow(2).sum()) \
                    * (int(sel.sum()) / n_pairs)
                loss = (data + reg) / n_pairs
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total += float(loss.detach())
            if log and (ep + 1) % 5 == 0:
                log(f"  zestxml epoch {ep + 1}/{epochs} objective {total:.4f}")
        return self

    @torch.no_grad()
    def scores(self, X, Y, cand, batch=256):
        """``alpha * bilinear + (1 - alpha) * knn``, zero outside the shortlist."""
        Yt, Wt = _to_torch(Y), self._Wt()
        cand = cand.tocsr()
        out = torch.zeros(X.shape[0], Y.shape[0])
        for lo in range(0, X.shape[0], batch):
            Xb = _dense(X[lo:lo + batch])
            sel = torch.as_tensor(np.asarray(cand[lo:lo + batch].todense()) != 0,
                                  device=DEVICE)
            m = self._margins(Xb, Yt, Wt) + self.b
            bil = torch.exp(-torch.clamp(1 - m, min=0) ** 2)     # margin -> positive score
            knn = self._margins(Xb, Yt, self.direct_t)
            out[lo:lo + batch] = ((self.alpha * bil + (1 - self.alpha) * knn)
                                  * sel).cpu()
        return out


def _dense(m):
    """A batch of points as a dense ``(batch, n_xf)`` tensor -- see :meth:`ZestXML._Wt`.

    At 256 x ~30k features this is ~30 MB. On a vocabulary ten times larger, drop
    ``batch`` rather than going sparse.
    """
    return torch.as_tensor(np.asarray(m.todense(), dtype=np.float32), device=DEVICE)


def _to_torch(m):
    m = sp.coo_matrix(m)
    return torch.sparse_coo_tensor(np.stack([m.row, m.col]), m.data.astype(np.float32),
                                   m.shape, device=DEVICE, dtype=torch.float32).coalesce()


def run_zestxml(split, ctx=20, windows=8, seed=0, epochs=20, alpha=0.9, shorty_k=500,
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

    # A training point is a prefix; its labels are the next ``horizon`` items -- the same
    # multi-label target the evaluation uses. Training on a single next item and then
    # scoring against a set would be a train/test mismatch, not a result.
    H = split.horizon
    trn_x, trn_rows, trn_cols = [], [], []
    for u, h in split.train.items():
        full = h + split.val[u]
        cuts = [k for k in range(1, len(full) - H + 1)]
        if not cuts:
            continue
        if len(cuts) > windows:
            cuts = sorted(rng.sample(cuts, windows))
        for k in cuts:
            r = len(trn_x)
            trn_x.append(doc(full[:k]))
            for it in sorted({i for i in full[k:k + H]}):
                trn_rows.append(r)
                trn_cols.append(split.index[it])

    tst_x = [doc(split.hist[u]) for u in split.users]

    vec = TfidfVectorizer(lowercase=True, token_pattern=r"[a-z][a-z0-9]+",
                          ngram_range=(1, 2), min_df=min_df, sublinear_tf=True,
                          stop_words="english")
    Xtr = _rownorm(vec.fit_transform(trn_x))
    Xte = _rownorm(vec.transform(tst_x))
    Xf = vec.get_feature_names_out().tolist()
    Y_full, Yf = label_features(names)
    Y_full = _rownorm(Y_full)

    XY = sp.csr_matrix((np.ones(len(trn_rows), np.float32), (trn_rows, trn_cols)),
                       shape=(len(trn_x), split.n_items))

    # THE zero-shot mechanism, and the easiest thing to get wrong: a label with no training
    # point has its features BLANKED for mining, shortlisting and training, and restored at
    # prediction. Leave them in during training and cold items get sampled as negatives --
    # the model learns to push down precisely what the cold column then asks it to rank up.
    Y_train = Y_full.copy().tolil()
    cold_rows = [split.index[i] for i in split.cold]
    Y_train[cold_rows, :] = 0
    Y_train = Y_train.tocsr()
    Y_train.eliminate_zeros()

    log(f"  {Xtr.shape[0]} points, {len(Xf)} point features, {len(Yf)} label features, "
        f"{len(cold_rows)} labels blanked for training")
    pattern, direct = mine_pattern(Xtr, Y_train, XY, Xf, Yf, bs_count, bs_alpha, direct_wt)
    log(f"  pattern nnz {pattern.nnz} ({direct.nnz} from the direct map)")

    trn_cand = shortlist(Xtr, Y_train, pattern, shorty_k)
    log(f"  train shortlist recall {100.0 * _recall(trn_cand, XY):.2f}%")
    tst_cand = shortlist(Xte, Y_full, pattern, shorty_k)
    log(f"  test shortlist recall  {100.0 * _recall(tst_cand, split.truth):.2f}%  "
        f"<- a hard ceiling on every metric below")

    model = ZestXML(pattern, direct, len(Xf), len(Yf), alpha=alpha)
    model.fit(Xtr, Y_train, XY, trn_cand, epochs=epochs, seed=seed, log=log)
    return model.scores(Xte, Y_full, tst_cand)


def _recall(cand, truth):
    hit = cand.multiply(truth)
    return hit.nnz / max(1, truth.nnz)


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

    # SASRec keeps its own next-item training signal -- that is what the architecture is
    # for, and handicapping it with a bag-of-items target would not make the comparison
    # fairer. Only the *evaluation* is shared, and that is where the framing has to match.
    users = sorted(split.train)
    seq = np.zeros((len(users), maxlen), np.int64)
    pos = np.zeros((len(users), maxlen), np.int64)
    for r, u in enumerate(users):
        h = [split.index[i] + 1 for i in split.train[u] + split.val[u]]   # 1-based, 0 pads
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
# 5b. controls  --  the rows that make the others readable
# =========================================================================== #
def run_popularity(split):
    """Rank every item by training frequency, identically for every user.

    Include this or the cold column is uninterpretable.  Group metrics mask every label
    outside the group, so the cold row asks "rank the 371 cold items", not "surface a cold
    item against the whole catalogue".  That question has a floor well above zero, and a
    model with no parameters for cold items can still beat uniform chance simply by
    ordering them in a fixed way that happens to put a test-frequent one first.  Any cold
    number that does not clear this row is measuring popularity, not transfer.
    """
    return torch.as_tensor(split.count, dtype=torch.float32)[None, :].repeat(
        len(split.users), 1)


def run_random(split, seed=0):
    """Uniform random scores -- the true chance floor for every column."""
    g = torch.Generator().manual_seed(seed)
    return torch.rand(len(split.users), split.n_items, generator=g)


# =========================================================================== #
# 6. runner
# =========================================================================== #
def main(dataset="ml-1m", cold_frac=0.1, horizon=5, sasrec_epochs=200, zest_epochs=20,
         ctx=20, windows=8, seed=0, shorty_k=500,
         arms=("random", "popularity", "zestxml", "sasrec", "sasrec+content"),
         **loader_kw):
    seqs, titles = LOADERS[dataset](**loader_kw)
    split = Split(seqs, titles, cold_frac=cold_frac, horizon=horizon, seed=seed)
    groups = split.groups()
    cold_pos = int(split.truth[:, groups["cold"]].sum())
    print(f"{len(split.train)} users, {split.n_items} items, {len(split.cold)} cold "
          f"({100.0 * len(split.cold) / split.n_items:.1f}%)")
    print(f"{len(split.users)} test points, {split.truth.nnz} positives "
          f"({split.truth.nnz / max(1, len(split.users)):.2f} labels per point), "
          f"{cold_pos} of them on cold items")

    rows = {}
    if "random" in arms:
        rows["random"] = evaluate(run_random(split, seed), split, groups)
    if "popularity" in arms:
        rows["popularity"] = evaluate(run_popularity(split), split, groups)
    if "zestxml" in arms:
        print("\n=== zestxml")
        rows["zestxml"] = evaluate(
            run_zestxml(split, ctx=ctx, windows=windows, seed=seed, epochs=zest_epochs,
                        shorty_k=shorty_k),
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
