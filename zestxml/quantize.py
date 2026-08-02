"""Residual-quantization k-means ("semantic IDs") over an embedding space.

An item is embedded once, then quantized at ``levels`` successive resolutions: k-means
with ``codebook_size`` centroids assigns a code, the centroid is subtracted, and the
residual is quantized again.  The item becomes ``levels`` discrete codes.  This is the
RQ-KMeans variant of the RQ-VAE semantic-ID idea (TIGER / OneRec), with plain k-means at
each level instead of a learned codebook.

The point of it here is that the codes can be emitted as *features*: a document gets the
point features ``rq0_<c0> ... rq{L-1}_<c>`` and a label the label features
``1_rq0_<c0> ...``.  ZestXML's direct map strips everything up to the first underscore
and tests string equality, so ``1_rq0_17`` links to ``rq0_17`` with no change to the
model at all -- a label and a document now share a feature when they land in the same
quantization cell rather than only when they share a literal word.

Two rules that this module enforces rather than leaves to the caller:

* the codebook is fit on whatever matrix you hand :func:`rq_kmeans`, and everything else
  is quantized through the returned ``transform`` closure, i.e. with the *same* fitted
  centroids.  Fitting on train+test is leakage.
* :func:`embed_texts` with ``require_all=True`` refuses to embed a name unless *every*
  one of its tokens has a vector.  Averaging over a partly out-of-vocabulary name
  collapses it onto its known tokens and manufactures spurious cosine-1.0 matches
  ("middleware webpack" == "middleware").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# embedding
# --------------------------------------------------------------------------- #
def simple_tokens(text: str) -> List[str]:
    """Lowercase alphanumeric tokenisation, matching the datasets' own token style."""
    out, cur = [], []
    for ch in text.lower():
        if ch.isalnum():
            cur.append(ch)
        else:
            if cur:
                out.append("".join(cur))
            cur = []
    if cur:
        out.append("".join(cur))
    return out


def embed_texts(
    texts: Sequence[str],
    index: Dict[str, int],
    vectors,
    require_all: bool = True,
    tokenizer: Callable[[str], List[str]] = simple_tokens,
    normalize: bool = True,
    max_tokens: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mean-of-word-vectors embedding.

    Returns ``(emb, keep)`` where ``keep`` are the indices of ``texts`` that could be
    embedded and ``emb`` has one row per kept item, in the order of ``keep``.

    ``require_all=True`` (the right setting for short *names*) drops any text with an
    out-of-vocabulary token.  ``require_all=False`` (the setting for long documents,
    where demanding full coverage would drop nearly everything) averages over the known
    tokens and drops only texts with no known token at all.
    """
    V = np.asarray(vectors, dtype=np.float32)
    rows, keep = [], []
    for i, text in enumerate(texts):
        toks = tokenizer(text)
        if max_tokens is not None:
            toks = toks[:max_tokens]
        if not toks:
            continue
        ids = [index[t] for t in toks if t in index]
        if require_all and len(ids) != len(toks):
            continue
        if not ids:
            continue
        v = V[ids].mean(axis=0)
        n = float(np.linalg.norm(v))
        if n == 0.0:
            continue
        rows.append(v / n if normalize else v)
        keep.append(i)
    if not rows:
        return np.zeros((0, V.shape[1]), dtype=np.float32), np.zeros(0, dtype=np.int64)
    return np.stack(rows).astype(np.float32), np.asarray(keep, dtype=np.int64)


# --------------------------------------------------------------------------- #
# residual k-means
# --------------------------------------------------------------------------- #
@dataclass
class LevelStat:
    level: int
    inertia: float
    used_codes: int
    codebook_size: int
    counts: np.ndarray  # items per code, over the fitting set (length codebook_size)
    residual_norm: float

    def line(self) -> str:
        nz = self.counts[self.counts > 0]
        return (
            "level %d: inertia %.2f, %d/%d codes used, items/code min %d med %d max %d, "
            "mean residual norm %.4f"
            % (
                self.level, self.inertia, self.used_codes, self.codebook_size,
                int(nz.min()) if nz.size else 0,
                int(np.median(nz)) if nz.size else 0,
                int(nz.max()) if nz.size else 0,
                self.residual_norm,
            )
        )


@dataclass
class RQCodebook:
    codebooks: List[np.ndarray]
    stats: List[LevelStat] = field(default_factory=list)

    @property
    def levels(self) -> int:
        return len(self.codebooks)

    def transform(self, X) -> np.ndarray:
        """Quantize new items with the already-fitted centroids -> codes (n, levels)."""
        from sklearn.metrics import pairwise_distances_argmin

        R = np.asarray(X, dtype=np.float32).copy()
        if R.ndim != 2:
            raise ValueError("expected a 2-D matrix")
        out = np.empty((R.shape[0], self.levels), dtype=np.int64)
        if R.shape[0] == 0:
            return out
        for l, C in enumerate(self.codebooks):
            c = pairwise_distances_argmin(R, C)
            out[:, l] = c
            R -= C[c]
        return out


def rq_kmeans(
    X,
    levels: int,
    codebook_size: int,
    seed: int = 0,
    minibatch: bool = False,
    n_init: int = 4,
    log=print,
) -> Tuple[RQCodebook, np.ndarray]:
    """Fit residual k-means on ``X``.  Returns ``(codebook, codes_of_X)``.

    ``X`` must be the *fitting* set only (for a dataset rewrite: the training
    documents).  Everything else goes through ``codebook.transform``.
    """
    from sklearn.cluster import KMeans, MiniBatchKMeans

    R = np.asarray(X, dtype=np.float32).copy()
    n = R.shape[0]
    if n == 0:
        raise ValueError("nothing to fit on")
    k = min(codebook_size, n)
    codebooks: List[np.ndarray] = []
    stats: List[LevelStat] = []
    codes = np.empty((n, levels), dtype=np.int64)
    for l in range(levels):
        if minibatch:
            km = MiniBatchKMeans(n_clusters=k, random_state=seed + l, n_init=n_init,
                                 batch_size=4096)
        else:
            km = KMeans(n_clusters=k, random_state=seed + l, n_init=n_init)
        km.fit(R)
        c = km.labels_.astype(np.int64)
        codes[:, l] = c
        C = km.cluster_centers_.astype(np.float32)
        codebooks.append(C)
        R -= C[c]
        counts = np.bincount(c, minlength=codebook_size)
        st = LevelStat(
            level=l, inertia=float(km.inertia_), used_codes=int((counts > 0).sum()),
            codebook_size=codebook_size, counts=counts,
            residual_norm=float(np.linalg.norm(R, axis=1).mean()),
        )
        stats.append(st)
        if log:
            log(st.line())
    return RQCodebook(codebooks=codebooks, stats=stats), codes


def sharing_report(codes: np.ndarray, levels: Optional[int] = None) -> List[str]:
    """Mean / max items per code, per level -- the diagnostic that predicts the sign."""
    out = []
    L = levels if levels is not None else codes.shape[1]
    for l in range(L):
        c = codes[:, l]
        if c.size == 0:
            out.append("level %d: no items" % l)
            continue
        counts = np.bincount(c)
        nz = counts[counts > 0]
        out.append(
            "level %d: %d distinct codes over %d items, %.2f items/code on average "
            "(max %d)" % (l, nz.size, c.size, float(nz.mean()), int(nz.max()))
        )
    return out
