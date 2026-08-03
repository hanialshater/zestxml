"""ZestXML against SASRec on the same sequential-recommendation task.

    python benchmarks/seqrec.py --data raw/ml-1m --cold_frac 0.1

The two models do not natively solve the same problem, so the comparison is only worth
anything if the mapping between them is stated rather than assumed:

* **SASRec** (Kang & McAuley, ICDM'18) reads a user's item-id sequence with a causal
  Transformer and scores the next item through a learned per-item embedding.  It sees the
  *order* of the history.  An item with no training interaction has an untrained embedding,
  so it can only be scored by accident.
* **ZestXML** reads one bag of tf-idf features per point and scores a label through the
  words of the label's name.  It throws the order away.  An item with no training
  interaction is still scored, through its title.

The mapping used here: one ZestXML *point* is a user history prefix, its text is the
concatenation of the titles and genres of the items in that prefix, and the *label* is the
next item, whose name is its own title and genres.  That is the same input and the same
target SASRec gets, minus the order -- which is precisely the trade being measured.

Everything else is held identical.  One split, one item catalogue, one held-out cold set,
one metric function, full-catalogue ranking, and the same history mask applied to both
score matrices.  The models are handed the same three tf-idf-derived content vectors too,
so the ``sasrec+content`` arm is not disadvantaged by having worse text than ZestXML.

**On published SASRec numbers.**  The familiar ml-1m figures (HR@10 around 0.82) are
measured against 100 sampled negatives, which Krichene & Rendle (KDD'20) showed is not a
consistent estimator of the full-catalogue metric.  Everything here ranks the full
catalogue, so these numbers are much lower and are *not* comparable to the published ones.
Do not put them in the same table.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ML1M_FILES = {
    "ratings.dat": "https://raw.githubusercontent.com/wesm/pydata-book/master/datasets/movielens/ratings.dat",
    "movies.dat": "https://raw.githubusercontent.com/wesm/pydata-book/master/datasets/movielens/movies.dat",
}


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def fetch_ml1m(root: str) -> str:
    """Download ml-1m if it is not already on disk.  Returns the directory."""
    import urllib.request

    os.makedirs(root, exist_ok=True)
    for name, url in ML1M_FILES.items():
        path = os.path.join(root, name)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            print(f"downloading {name} ...")
            urllib.request.urlretrieve(url, path)
    return root


def load_ml1m(root: str, min_len: int = 5) -> Tuple[Dict[int, List[int]], Dict[int, str]]:
    """Return ``({user: [item, ...] in time order}, {item: "title genres"})``.

    ml-1m ships one rating per line and no explicit sequence; the sequence is the ratings
    sorted by timestamp, which is what every SASRec paper uses.  Users with fewer than
    ``min_len`` interactions are dropped, also standard -- leave-one-out needs a train
    prefix, a validation item and a test item to exist at all.
    """
    titles: Dict[int, str] = {}
    with open(os.path.join(root, "movies.dat"), encoding="latin-1") as f:
        for line in f:
            parts = line.rstrip("\n").split("::")
            if len(parts) >= 3:
                titles[int(parts[0])] = f"{parts[1]} {parts[2].replace('|', ' ')}"

    events: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    with open(os.path.join(root, "ratings.dat"), encoding="latin-1") as f:
        for line in f:
            p = line.rstrip("\n").split("::")
            if len(p) >= 4:
                events[int(p[0])].append((int(p[3]), int(p[1])))

    seqs = {}
    for u, rows in events.items():
        rows.sort()  # by timestamp, then item id -- deterministic under ties
        items = [i for _, i in rows if i in titles]
        if len(items) >= min_len:
            seqs[u] = items
    return seqs, titles


@dataclass
class Split:
    """One leave-one-out split with a held-out cold item set."""

    train: Dict[int, List[int]]          # user -> history used for training
    val_target: Dict[int, int]
    test_target: Dict[int, int]
    test_history: Dict[int, List[int]]   # train + val item, the history at test time
    items: List[int]                     # catalogue, in index order
    item_index: Dict[int, int]
    titles: Dict[int, str]
    cold: set = field(default_factory=set)
    train_count: Optional[np.ndarray] = None

    @property
    def n_items(self) -> int:
        return len(self.items)


def make_split(seqs, titles, cold_frac: float = 0.1, seed: int = 0) -> Split:
    """Leave-one-out split, then remove a fraction of items from training entirely.

    The cold construction is the point of the exercise, so it is worth being precise about
    it: an item is chosen cold, then *every* occurrence of it is deleted from every training
    history and from the validation targets.  It survives only where it is somebody's final
    test target.  That leaves it with exactly zero training interactions -- not few, zero --
    which is the condition SASRec's item embedding cannot be trained under and ZestXML's
    label text can.
    """
    rng = random.Random(seed)
    items = sorted({i for s in seqs.values() for i in s})
    index = {it: k for k, it in enumerate(items)}

    cold = set()
    if cold_frac > 0:
        n = int(round(cold_frac * len(items)))
        # only items that are somebody's last interaction can be held out and still be
        # measurable -- a cold item nobody is tested on contributes nothing but noise
        last_items = {s[-1] for s in seqs.values() if len(s) >= 3}
        pool = sorted(last_items)
        rng.shuffle(pool)
        cold = set(pool[:n])

    train, val_t, test_t, test_h = {}, {}, {}, {}
    for u, s in seqs.items():
        if len(s) < 3:
            continue
        hist, val, tst = s[:-2], s[-2], s[-1]
        hist = [i for i in hist if i not in cold]
        if len(hist) < 2:
            continue
        train[u] = hist
        if val not in cold:
            val_t[u] = val
        test_t[u] = tst
        test_h[u] = hist + ([val] if val not in cold else [])

    count = np.zeros(len(items), dtype=np.int64)
    for h in train.values():
        for i in h:
            count[index[i]] += 1
    for u, v in val_t.items():
        count[index[v]] += 1

    return Split(train, val_t, test_t, test_h, items, index, titles, cold, count)


def frequency_groups(split: Split, head_frac: float = 0.2) -> Dict[str, np.ndarray]:
    """Boolean item masks for head / tail / cold, by training frequency."""
    c = split.train_count
    warm = np.nonzero(c > 0)[0]
    order = warm[np.argsort(-c[warm])]
    cut = int(round(head_frac * len(order)))
    head = np.zeros(split.n_items, dtype=bool)
    head[order[:cut]] = True
    tail = (c > 0) & ~head
    cold = c == 0
    return {"head": head, "tail": tail, "cold": cold}


# --------------------------------------------------------------------------- #
# shared evaluation
# --------------------------------------------------------------------------- #
def evaluate(scores: torch.Tensor, targets: np.ndarray, history: Sequence[Sequence[int]],
             groups: Dict[str, np.ndarray], ks=(10, 50)) -> Dict[str, Dict[str, float]]:
    """One metric function for both models.

    ``scores`` is ``(n_users, n_items)`` dense, ``targets`` the item index each row should
    retrieve.  Items already in a user's history are masked out, which is the standard
    protocol and matters here because ZestXML, having no notion of a sequence, would
    otherwise happily re-recommend what the user just watched.
    """
    scores = scores.clone().float()
    for r, h in enumerate(history):
        if len(h):
            scores[r, torch.as_tensor(h, dtype=torch.long)] = -np.inf

    top = torch.topk(scores, max(ks), dim=1).indices.numpy()
    tgt = np.asarray(targets)
    hit_at = (top == tgt[:, None])
    rank = np.where(hit_at.any(1), hit_at.argmax(1), max(ks) + 1)  # 0-based, or "missed"

    out: Dict[str, Dict[str, float]] = {}
    for name, mask in [("all", np.ones(len(tgt), dtype=bool))] + [
            (g, m[tgt]) for g, m in groups.items()]:
        if mask.sum() == 0:
            continue
        r = rank[mask]
        row = {"n": int(mask.sum())}
        for k in ks:
            row[f"Recall@{k}"] = 100.0 * float((r < k).mean())
        row["NDCG@10"] = 100.0 * float(
            np.where(r < 10, 1.0 / np.log2(np.asarray(r, dtype=float) + 2), 0.0).mean())
        row["MRR"] = 100.0 * float(np.where(r <= max(ks), 1.0 / (r + 1.0), 0.0).mean())
        out[name] = row
    # catalogue coverage: how much of the catalogue the model ever puts in a top-10
    out["all"]["coverage@10"] = 100.0 * len(np.unique(top[:, :10])) / scores.shape[1]
    return out


def format_table(rows: Dict[str, Dict[str, Dict[str, float]]]) -> str:
    cols = ["Recall@10", "Recall@50", "NDCG@10", "MRR"]
    lines = ["%-22s %-6s %6s " % ("model", "group", "n")
             + " ".join("%9s" % c for c in cols)]
    for model, groups in rows.items():
        for g, m in groups.items():
            lines.append("%-22s %-6s %6d " % (model, g, m["n"])
                         + " ".join("%9.2f" % m[c] for c in cols))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# SASRec
# --------------------------------------------------------------------------- #
class SASRec(nn.Module):
    """Kang & McAuley's causal self-attention recommender, compactly.

    ``content`` optionally adds a fixed per-item content vector through a learned linear
    map, so an item with no training interaction still has a non-random representation.
    Without it the cold-start column measures "an untrained embedding table", which is a
    true statement about SASRec but a weak comparison; with it, the cold column measures
    what a practitioner would actually deploy.
    """

    def __init__(self, n_items: int, d: int = 50, maxlen: int = 200, blocks: int = 2,
                 heads: int = 1, dropout: float = 0.2, content: Optional[torch.Tensor] = None):
        super().__init__()
        self.n_items, self.maxlen = n_items, maxlen
        self.item = nn.Embedding(n_items + 1, d, padding_idx=0)  # 0 is the pad slot
        self.pos = nn.Embedding(maxlen, d)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(d, heads, d * 4, dropout, batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, blocks, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.content_proj = None
        if content is not None:
            self.register_buffer("content", content)
            self.content_proj = nn.Linear(content.shape[1], d, bias=False)
        nn.init.normal_(self.item.weight, std=0.02)
        with torch.no_grad():
            self.item.weight[0].zero_()

    def item_vectors(self) -> torch.Tensor:
        """(n_items + 1, d) -- the id embedding plus the projected content, if any."""
        v = self.item.weight
        if self.content_proj is not None:
            extra = torch.zeros_like(v)
            extra[1:] = self.content_proj(self.content)
            v = v + extra
        return v

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """``seq`` is (B, maxlen) of 1-based item ids, 0-padded on the left."""
        n = seq.shape[1]
        table = self.item_vectors()
        x = table[seq] * (table.shape[1] ** 0.5)
        x = self.drop(x + self.pos(torch.arange(n, device=seq.device))[None])
        x = x * (seq > 0).unsqueeze(-1)
        causal = torch.triu(torch.ones(n, n, dtype=torch.bool, device=seq.device), 1)
        x = self.encoder(x, mask=causal, src_key_padding_mask=(seq == 0))
        return self.norm(torch.nan_to_num(x))


def _windows(split: Split, maxlen: int) -> Tuple[np.ndarray, np.ndarray]:
    """One left-padded training window per user, with the next-item target at each step."""
    users = sorted(split.train)
    seq = np.zeros((len(users), maxlen), dtype=np.int64)
    pos = np.zeros((len(users), maxlen), dtype=np.int64)
    for r, u in enumerate(users):
        # ids are 1-based inside the model so that 0 can be the pad slot
        h = [split.item_index[i] + 1 for i in split.train[u]]
        if u in split.val_target:
            h = h + [split.item_index[split.val_target[u]] + 1]
        h = h[-(maxlen + 1):]
        inp, tgt = h[:-1], h[1:]
        seq[r, maxlen - len(inp):] = inp
        pos[r, maxlen - len(tgt):] = tgt
    return seq, pos


def train_sasrec(split: Split, content: Optional[np.ndarray] = None, d: int = 50,
                 maxlen: int = 200, epochs: int = 200, lr: float = 1e-3,
                 batch_size: int = 128, device=None, seed: int = 0, log=print) -> SASRec:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    ct = None if content is None else torch.as_tensor(content, dtype=torch.float32, device=device)
    model = SASRec(split.n_items, d=d, maxlen=maxlen, content=ct).to(device)
    seq, pos = _windows(split, maxlen)
    seq_t = torch.as_tensor(seq, device=device)
    pos_t = torch.as_tensor(pos, device=device)
    # negatives are drawn from items with at least one training interaction: sampling a
    # cold item as a negative would train the model to push down exactly the items the
    # cold-start column then asks it to rank up, which is a rigged comparison
    warm = torch.as_tensor(np.nonzero(split.train_count > 0)[0] + 1, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98))
    gen = torch.Generator(device="cpu").manual_seed(seed)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    for epoch in range(epochs):
        model.train()
        order = torch.randperm(seq_t.shape[0], generator=gen).to(device)
        total, seen = 0.0, 0
        for lo in range(0, order.numel(), batch_size):
            b = order[lo:lo + batch_size]
            state = model(seq_t[b])
            table = model.item_vectors()
            p = pos_t[b]
            neg = warm[torch.randint(warm.numel(), p.shape, device=device)]
            mask = p > 0
            pos_logit = (state * table[p]).sum(-1)
            neg_logit = (state * table[neg]).sum(-1)
            loss = (bce(pos_logit, torch.ones_like(pos_logit))
                    + bce(neg_logit, torch.zeros_like(neg_logit)))
            loss = (loss * mask).sum() / mask.sum().clamp_min(1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * int(mask.sum())
            seen += int(mask.sum())
        if log and (epoch + 1) % 20 == 0:
            log(f"  sasrec epoch {epoch + 1}/{epochs} loss {total / max(1, seen):.4f}")
    return model


@torch.no_grad()
def score_sasrec(model: SASRec, split: Split, maxlen: int = 200, device=None) -> torch.Tensor:
    device = device or next(model.parameters()).device
    model.eval()
    users = sorted(split.test_target)
    seq = np.zeros((len(users), maxlen), dtype=np.int64)
    for r, u in enumerate(users):
        h = [split.item_index[i] + 1 for i in split.test_history[u]][-maxlen:]
        seq[r, maxlen - len(h):] = h
    out = torch.zeros(len(users), split.n_items)
    table = model.item_vectors()[1:]  # drop the pad row
    for lo in range(0, len(users), 256):
        b = torch.as_tensor(seq[lo:lo + 256], device=device)
        state = model(b)[:, -1]  # the representation after the last real item
        out[lo:lo + 256] = (state @ table.T).cpu()
    return out


# --------------------------------------------------------------------------- #
# ZestXML side
# --------------------------------------------------------------------------- #
def item_names(split: Split) -> List[str]:
    """Label names, one per item, forced unique.

    ``build_dataset`` maps documents to labels *by name*, so two items sharing a title
    would silently merge.  ml-1m titles carry their year and collide rarely; the few that
    do get a disambiguating token, and the count is reported rather than hidden.
    """
    names, seen, clashes = [], {}, 0
    for it in split.items:
        base = split.titles.get(it, f"item {it}")
        if base in seen:
            clashes += 1
            base = f"{base} v{seen[base] + 1}"
        seen[base] = seen.get(base, 0) + 1
        names.append(base)
    if clashes:
        print(f"note: {clashes} duplicate titles were disambiguated")
    return names


def build_zx_dataset(split: Split, out_dir: str, windows_per_user: int = 8,
                     ctx: int = 20, seed: int = 0, min_df: int = 2) -> Dict:
    """Write the sequential task as a GZXML dataset.

    A training point is a prefix of a user's history and its label is the next item, so one
    user yields several points -- the same signal SASRec gets from predicting at every
    position, with the order discarded.  ``ctx`` bounds how many recent items make up the
    text: a 200-item history averaged into one tf-idf vector is mostly noise, and the
    recency cut is the only piece of sequence information the bag-of-words model keeps.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    from zestxml.dataset import build_dataset, one_line

    rng = random.Random(seed)
    names = item_names(split)
    text_of = {it: split.titles.get(it, "") for it in split.items}

    def doc(prefix: Sequence[int]) -> str:
        return one_line(" ".join(text_of[i] for i in prefix[-ctx:]))

    trn_x, trn_y = [], []
    for u, hist in split.train.items():
        full = hist + ([split.val_target[u]] if u in split.val_target else [])
        positions = list(range(1, len(full)))
        if len(positions) > windows_per_user:
            positions = sorted(rng.sample(positions, windows_per_user))
        for k in positions:
            trn_x.append(doc(full[:k]))
            trn_y.append([names[split.item_index[full[k]]]])

    users = sorted(split.test_target)
    tst_x = [doc(split.test_history[u]) for u in users]
    tst_y = [[names[split.item_index[split.test_target[u]]]] for u in users]

    unseen = {split.item_index[i] for i in split.cold}
    return build_dataset(
        out_dir, trn_x, trn_y, tst_x, tst_y, label_names=names, unseen=unseen,
        vectorizer=TfidfVectorizer(lowercase=True, token_pattern=r"[a-z][a-z0-9]+",
                                   ngram_range=(1, 2), min_df=min_df, sublinear_tf=True,
                                   stop_words="english"),
    )


def content_vectors(split: Split, dim: int = 128, seed: int = 0) -> np.ndarray:
    """A fixed content vector per item, from the same text ZestXML reads.

    Deliberately not a pretrained sentence encoder: the point of the ``+content`` arm is to
    give SASRec the *same* information ZestXML has, so that a difference between them is
    about the model and not about who got the better text features.
    """
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = [split.titles.get(i, "") for i in split.items]
    tf = TfidfVectorizer(lowercase=True, token_pattern=r"[a-z][a-z0-9]+",
                         ngram_range=(1, 2), min_df=1, sublinear_tf=True,
                         stop_words="english").fit_transform(texts)
    dim = min(dim, tf.shape[1] - 1)
    v = TruncatedSVD(dim, random_state=seed).fit_transform(tf).astype(np.float32)
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-8)


def zx_scores(split: Split, data_dir: str, res_dir: str, **cfg) -> torch.Tensor:
    from zestxml import ZestXML
    from zestxml.io import read_bin_smat

    config = dict(bs_alpha=0.02, bs_direct_wt=0.8, bilinear_classifier_cost=5,
                  bilinear_normalize=0, bilinear_classifier_maxitr=20, num_thread=0)
    config.update(cfg)
    model = ZestXML(data_dir, res_dir, **config)
    model.fit()
    model.predict()
    return read_bin_smat(f"{res_dir}/score_mat.bin").to_dense().cpu()


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #
def main(data="raw/ml-1m", out="Results/SeqRec", cold_frac=0.1, epochs=200, ctx=20,
         windows=8, maxlen=200, seed=0, arms=("zestxml", "sasrec", "sasrec+content"),
         **cfg):
    fetch_ml1m(data)
    seqs, titles = load_ml1m(data)
    split = make_split(seqs, titles, cold_frac=cold_frac, seed=seed)
    groups = frequency_groups(split)
    users = sorted(split.test_target)
    targets = np.array([split.item_index[split.test_target[u]] for u in users])
    history = [[split.item_index[i] for i in split.test_history[u]] for u in users]
    print(f"{len(split.train)} users, {split.n_items} items, {len(split.cold)} cold "
          f"({100.0 * len(split.cold) / split.n_items:.1f}%), "
          f"{len(users)} test points, {int((groups['cold'][targets]).sum())} of them cold")

    rows: Dict[str, Dict] = {}
    if "zestxml" in arms:
        print("\n=== ZestXML")
        ds = f"GZXML-Datasets/SeqRec-ml1m-c{cold_frac}"
        build_zx_dataset(split, ds, windows_per_user=windows, ctx=ctx, seed=seed)
        s = zx_scores(split, ds, f"{out}/zestxml", **cfg)
        rows["zestxml"] = evaluate(s, targets, history, groups)

    content = content_vectors(split, seed=seed)
    for name, ct in (("sasrec", None), ("sasrec+content", content)):
        if name not in arms:
            continue
        print(f"\n=== {name}")
        m = train_sasrec(split, content=ct, maxlen=maxlen, epochs=epochs, seed=seed)
        rows[name] = evaluate(score_sasrec(m, split, maxlen=maxlen), targets, history, groups)

    print("\n" + "=" * 78)
    print(format_table(rows))
    print("\ncoverage@10: " + ", ".join(
        f"{k} {v['all']['coverage@10']:.1f}%" for k, v in rows.items()))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="raw/ml-1m")
    ap.add_argument("--out", default="Results/SeqRec")
    ap.add_argument("--cold_frac", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--ctx", type=int, default=20, help="recent items making up a point's text")
    ap.add_argument("--windows", type=int, default=8, help="training prefixes per user")
    ap.add_argument("--maxlen", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arms", default="zestxml,sasrec,sasrec+content")
    a = ap.parse_args()
    main(a.data, a.out, a.cold_frac, a.epochs, a.ctx, a.windows, a.maxlen, a.seed,
         tuple(a.arms.split(",")))
