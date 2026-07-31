"""Fuzzy matching between label-feature names and point-feature names.

``create_Xf_Yf_map_direct`` links a label feature to a point feature only when their
names are *string equal*. That is the one bridge an unseen label has to the document
vocabulary, and it is brittle: ``acq`` never reaches ``acquisition``, ``nat-gas`` never
reaches ``natural gas``, and a label whose name is absent from the vocabulary is
unreachable by construction (10% of the npm keywords, 17% of the Reuters topics).

This module replaces equality with a similarity search over the feature names:

* ``charngram`` -- tf-idf over character n-grams, computed from the vocabularies
  themselves. Needs no downloads and catches morphology, compounds and misspellings.
* ``vectors``   -- cosine similarity in a supplied word-vector space (word2vec/GloVe/
  fastText text format), which additionally catches synonymy. Multi-word feature names
  are averaged.

Both return the same thing: for every label feature, its ``topk`` closest point features
with a cosine similarity, which the caller turns into weighted links.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from .csr import CSR, bounded_chunks, prod_dense_rows, row_costs

CHAR_NGRAMS = (3, 5)


# --------------------------------------------------------------------------- #
# character n-grams
# --------------------------------------------------------------------------- #
def _ngrams(text: str, lo: int, hi: int) -> List[str]:
    padded = "<" + text.strip().lower() + ">"
    out = []
    for n in range(lo, hi + 1):
        out.extend(padded[i : i + n] for i in range(max(0, len(padded) - n + 1)))
    return out


def char_ngram_matrices(
    left: Sequence[str], right: Sequence[str], ngrams: Tuple[int, int] = CHAR_NGRAMS,
    device=None, dtype=torch.float32,
) -> Tuple[CSR, CSR]:
    """tf-idf character n-gram rows for both name lists, sharing one vocabulary."""
    vocab: Dict[str, int] = {}
    docs: List[List[int]] = []
    for names in (left, right):
        for name in names:
            ids = []
            for g in _ngrams(name, *ngrams):
                ids.append(vocab.setdefault(g, len(vocab)))
            docs.append(ids)

    n_docs = len(docs)
    df = torch.zeros(len(vocab))
    for ids in docs:
        df[torch.tensor(sorted(set(ids)), dtype=torch.long)] += 1
    idf = torch.log((n_docs + 1) / (df + 1)) + 1.0

    def build(slice_docs, shape_rows):
        rows, cols = [], []
        for i, ids in enumerate(slice_docs):
            rows.extend([i] * len(ids))
            cols.extend(ids)
        rows_t = torch.tensor(rows, dtype=torch.long)
        cols_t = torch.tensor(cols, dtype=torch.long)
        vals = idf[cols_t] if cols_t.numel() else torch.zeros(0)
        mat = CSR.from_coo(rows_t, cols_t, vals.to(dtype), (shape_rows, len(vocab)))
        # repeated n-grams within a name accumulate, which is the tf part
        return mat.unit_normalize_rows().to(device)

    return build(docs[: len(left)], len(left)), build(docs[len(left) :], len(right))


# --------------------------------------------------------------------------- #
# word vectors
# --------------------------------------------------------------------------- #
def load_word_vectors(path: str, dtype=torch.float32) -> Tuple[Dict[str, int], Tensor]:
    """Read a word2vec/GloVe/fastText *text* format file."""
    index: Dict[str, int] = {}
    rows: List[List[float]] = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        first = f.readline().split()
        if len(first) > 2:  # no header line: it was already a vector
            index[first[0]] = 0
            rows.append([float(x) for x in first[1:]])
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) < 3:
                continue
            index[parts[0]] = len(rows)
            rows.append([float(x) for x in parts[1:]])
    return index, torch.tensor(rows, dtype=dtype)


def word_vector_matrices(
    left: Sequence[str], right: Sequence[str], path: str, device=None, dtype=torch.float32
) -> Tuple[Tensor, Tensor]:
    """Average the vectors of each name's tokens, L2 normalised. Unknown names get 0."""
    index, table = load_word_vectors(path, dtype)

    def embed(names):
        out = torch.zeros(len(names), table.shape[1], dtype=dtype)
        for i, name in enumerate(names):
            ids = [index[t] for t in name.lower().replace("-", " ").split() if t in index]
            if ids:
                out[i] = table[torch.tensor(ids, dtype=torch.long)].mean(0)
        return torch.nn.functional.normalize(out, dim=1).to(device)

    return embed(left), embed(right)


# --------------------------------------------------------------------------- #
# top-k cosine search
# --------------------------------------------------------------------------- #
def topk_cosine_sparse(
    left: CSR, right: CSR, topk: int, min_sim: float,
    max_elems: int = 1 << 26, dense_elems: int = 1 << 24,
) -> CSR:
    """Top-``topk`` columns of ``left @ right^T`` above ``min_sim``, rows L2 normalised."""
    right_t = right.transpose()
    n_right = right.nrows
    costs = row_costs(left, right_t)
    max_rows = max(1, dense_elems // max(1, n_right))

    keys, vals = [], []
    for lo, hi in bounded_chunks(costs, max_elems, max_rows):
        rows = torch.arange(lo, hi, device=left.device)
        block = prod_dense_rows(left, right_t, rows)
        k = min(topk, n_right)
        top_val, top_idx = torch.topk(block, k, dim=1)
        keep = top_val >= min_sim
        if not bool(keep.any()):
            continue
        idx = keep.nonzero(as_tuple=False)
        keys.append((idx[:, 0] + lo) * n_right + top_idx[keep])
        vals.append(top_val[keep])

    if not keys:
        return CSR.empty((left.nrows, n_right), device=left.device, dtype=left.values.dtype)
    keys_t, vals_t = torch.cat(keys), torch.cat(vals)
    order = torch.argsort(keys_t)
    return CSR.from_sorted_keys(keys_t[order], vals_t[order], (left.nrows, n_right))


def topk_cosine_dense(left: Tensor, right: Tensor, topk: int, min_sim: float, chunk: int = 1024) -> CSR:
    keys, vals = [], []
    n_right = right.shape[0]
    k = min(topk, n_right)
    for lo in range(0, left.shape[0], chunk):
        block = left[lo : lo + chunk] @ right.t()
        top_val, top_idx = torch.topk(block, k, dim=1)
        keep = top_val >= min_sim
        if not bool(keep.any()):
            continue
        idx = keep.nonzero(as_tuple=False)
        keys.append((idx[:, 0] + lo) * n_right + top_idx[keep])
        vals.append(top_val[keep])
    if not keys:
        return CSR.empty((left.shape[0], n_right), device=left.device, dtype=left.dtype)
    keys_t, vals_t = torch.cat(keys), torch.cat(vals)
    order = torch.argsort(keys_t)
    return CSR.from_sorted_keys(keys_t[order], vals_t[order], (left.shape[0], n_right))


def similar_names(
    queries: Sequence[str], targets: Sequence[str], mode: str, topk: int = 3, min_sim: float = 0.5,
    vectors: Optional[str] = None, device=None, dtype=torch.float32,
    max_elems: int = 1 << 26, dense_elems: int = 1 << 24,
) -> CSR:
    """(len(queries) x len(targets)) matrix of cosine similarities, top-``topk`` per query."""
    if mode == "charngram":
        left, right = char_ngram_matrices(queries, targets, device=device, dtype=dtype)
        return topk_cosine_sparse(left, right, topk, min_sim, max_elems, dense_elems)
    if mode == "vectors":
        if not vectors:
            raise ValueError("-direct_map vectors needs -direct_vectors <path>")
        left, right = word_vector_matrices(queries, targets, vectors, device=device, dtype=dtype)
        return topk_cosine_dense(left, right, topk, min_sim)
    raise ValueError(f"unknown direct_map mode: {mode}")
