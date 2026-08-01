"""Stage 1: mine the sparsity pattern of the bilinear matrix W (num_Xf x num_Yf).

Port of ``create_Xf_Yf`` / ``create_Xf_Yf_map`` / ``create_Xf_Yf_map_direct`` /
``remove_duplicates`` / ``prod_for_jaccard`` from the C++ reference implementation.

Two feature-pair scores are combined:

* a co-occurrence score between a point feature ``xf`` and a label feature ``yf``,
  counted over training (point, label) pairs and normalised by a weighted mix of a
  Jaccard denominator and the product of marginal frequencies::

      s(xf, yf) = c(xf, yf) / (alpha * (f(yf) + f(xf) - c(xf, yf))
                               + (1 - alpha) * g(xf) * g(yf))

  Every ``xf`` keeps its ``bs_count`` best ``yf`` (``Xf_Yf``) and, symmetrically, every
  ``yf`` keeps its ``bs_count`` best ``xf`` (``Yf_Xf``).
* a "direct" score with fixed weight ``bs_direct_wt`` linking a label feature to the
  identically named point feature, which is what lets an unseen label's tokens reach the
  document vocabulary at all.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from .csr import CSR, bounded_chunks, prod_dense_rows, row_costs, spspmm, topk_per_row_dense

GOOD_TH = 100.0  # prod_for_jaccard keeps entries above this on top of the top-k


def _jaccard_topk(
    left: CSR,
    right: CSR,
    row_freq: Tensor,
    col_freq: Tensor,
    row_afreq: Tensor,
    col_afreq: Tensor,
    alpha: float,
    k: int,
    threshold: float,
    max_elems: int,
    dense_elems: int,
) -> CSR:
    """``left @ right`` scored with the formula above and truncated to top-``k`` per row."""
    ncols = right.ncols
    costs = row_costs(left, right)
    max_rows = max(1, dense_elems // max(1, ncols))
    keys_out: List[Tensor] = []
    vals_out: List[Tensor] = []

    for lo, hi in bounded_chunks(costs, max_elems, max_rows):
        rows = torch.arange(lo, hi, device=left.device)
        block = prod_dense_rows(left, right, rows)
        denom = alpha * (row_freq[lo:hi, None] + col_freq[None, :] - block) + (1.0 - alpha) * (
            row_afreq[lo:hi, None] * col_afreq[None, :]
        )
        block = torch.where(block > 0, block / denom.clamp_min(1e-12), torch.zeros_like(block))
        del denom

        keep = topk_per_row_dense(block, k, GOOD_TH)
        # SMat::threshold, then the explicit "value -= threshold" loop in create_Xf_Yf_map
        if threshold > 0:
            keep &= block.abs() > threshold
        idx = keep.nonzero(as_tuple=False)
        if idx.numel() == 0:
            continue
        keys_out.append((idx[:, 0] + lo) * ncols + idx[:, 1])
        vals_out.append(block[keep] - threshold)

    if not keys_out:
        return CSR.empty((left.nrows, ncols), device=left.device, dtype=left.values.dtype)
    return CSR.from_sorted_keys(torch.cat(keys_out), torch.cat(vals_out), (left.nrows, ncols))


def _query_name(name: str) -> str:
    """The part of a label-feature name that is looked up in the point vocabulary.

    Label features carry a prefix up to the first underscore; whatever follows is the
    lookup key. Names without an underscore are matched whole, which is what the C++
    ``substr(npos + 1)`` does.
    """
    return name[name.find("_") + 1 :]


def _is_token_feature(query: str) -> bool:
    """Whether a label feature names a token rather than a label.

    The per-label ``__label__<i>__<name>`` features strip to ``_label__<i>__<name>``,
    which never matches anything exactly; under fuzzy matching they would match plenty of
    nonsense, so they are excluded from the similarity search.
    """
    return bool(query) and not query.startswith("_") and "__" not in query


def direct_map(
    Xf: List[str], Yf: List[str], weight: float, device=None, dtype=torch.float32,
    mode: str = "exact", topk: int = 3, min_sim: float = 0.5, vectors: Optional[str] = None,
    fallback_only: bool = True,
    max_elems: int = 1 << 26, dense_elems: int = 1 << 24, log=print,
) -> CSR:
    """``create_Xf_Yf_map_direct``: link ``yf`` to the point feature it is named after.

    With ``mode='exact'`` this is the reference behaviour. Otherwise each token-like label
    feature also links to its ``topk`` nearest point features under
    :mod:`zestxml.embed`, weighted by ``weight * cosine`` so a loose match counts for
    less than an exact one.
    """
    xf_index: Dict[str, int] = {}
    for i, name in enumerate(Xf):
        xf_index[name] = i  # a later duplicate wins, as with std::map assignment

    rows: List[int] = []
    cols: List[int] = []
    for j, name in enumerate(Yf):
        i = xf_index.get(_query_name(name))
        if i is not None:
            rows.append(i)
            cols.append(j)

    exact = CSR.from_coo(
        torch.tensor(rows, dtype=torch.long, device=device),
        torch.tensor(cols, dtype=torch.long, device=device),
        torch.full((len(rows),), weight, dtype=dtype, device=device),
        (len(Xf), len(Yf)),
    )
    if mode == "exact":
        return exact

    from .embed import similar_names

    queries = [_query_name(name) for name in Yf]
    keep = [j for j, q in enumerate(queries) if _is_token_feature(q)]
    if fallback_only:
        # an exact match is high precision; only look for neighbours when there is none
        matched = set(cols)
        keep = [j for j in keep if j not in matched]
    log(f"fuzzy direct map ({mode}, {'fallback' if fallback_only else 'augment'}): "
        f"{len(keep)}/{len(Yf)} label features, top {topk} above {min_sim}")
    if not keep:
        return exact
    sims = similar_names(
        [queries[j] for j in keep], Xf, mode, topk=topk, min_sim=min_sim, vectors=vectors,
        device=device, dtype=dtype, max_elems=max_elems, dense_elems=dense_elems, log=log,
    )

    keep_t = torch.tensor(keep, dtype=torch.long, device=device)
    fuzzy_rows = sims.indices  # point features
    fuzzy_cols = keep_t[sims.row_ids()]  # label features
    fuzzy_vals = sims.values * weight

    # one link per pair: an exact match always outranks a fuzzy one of the same pair
    all_rows = torch.cat([exact.row_ids(), fuzzy_rows])
    all_cols = torch.cat([exact.indices, fuzzy_cols])
    all_vals = torch.cat([exact.values, fuzzy_vals])
    keys = all_rows * len(Yf) + all_cols
    order = torch.argsort(keys)
    keys, all_vals = keys[order], all_vals[order]
    uniq, inv = torch.unique(keys, return_inverse=True)
    best = torch.zeros(uniq.numel(), dtype=dtype, device=device).scatter_reduce(
        0, inv, all_vals, reduce="amax", include_self=False
    )
    log(f"fuzzy direct map: {exact.nnz} exact + {uniq.numel() - exact.nnz} new links")
    return CSR.from_sorted_keys(uniq, best, (len(Xf), len(Yf)))


def add_matrices(a: CSR, b: CSR) -> CSR:
    """``SMat::add``: sum the two matrices, keeping ``a``'s entry order per row.

    Entries of ``b`` that ``a`` already has are added in place; the rest are appended
    after ``a``'s entries in that row. Reproducing this ordering matters because the
    order of ``Xf_Yf``'s non-zeros *is* the layout of the learnt parameter vector.
    """
    assert a.shape == b.shape, f"shape mismatch: {a.shape} vs {b.shape}"
    b_keys = b.keys()
    if b.nnz and not bool((b_keys[1:] >= b_keys[:-1]).all()):
        b = b.sort_indices()  # searchsorted below needs b's keys ascending
        b_keys = b.keys()
    a_keys = a.keys()
    values = a.values.clone()

    if b.nnz and a.nnz:
        pos = torch.searchsorted(b_keys, a_keys).clamp(max=b.nnz - 1)
        hit = b_keys[pos] == a_keys
        values[hit] += b.values[pos[hit]]
        fresh = ~torch.isin(b_keys, a_keys)
    else:
        fresh = torch.ones(b.nnz, dtype=torch.bool, device=b.device)

    n_rows, n_cols = a.shape
    rows = torch.cat([a.row_ids(), b.row_ids()[fresh]])
    cols = torch.cat([a.indices, b.indices[fresh]])
    vals = torch.cat([values, b.values[fresh]])
    tail = torch.cat(
        [torch.zeros(a.nnz, dtype=torch.long, device=a.device), torch.ones(int(fresh.sum()), dtype=torch.long, device=a.device)]
    )
    order = torch.argsort((rows * 2 + tail) * n_cols + cols)
    return CSR.from_coo(rows[order], cols[order], vals[order], (n_rows, n_cols), sort=False)


def remove_duplicates(Yf_Xf: CSR, Xf_Yf: CSR) -> CSR:
    """Drop ``(yf, xf)`` entries whose transpose already lives in ``Xf_Yf``."""
    transposed_keys = Yf_Xf.indices * Xf_Yf.ncols + Yf_Xf.row_ids()
    return Yf_Xf.subset(~torch.isin(transposed_keys, Xf_Yf.keys()))


def build_sparsity_pattern(
    trn_X_Xf: CSR,
    Y_Yf: CSR,
    trn_X_Y: CSR,
    Xf: List[str],
    Yf: List[str],
    bs_count: int = 10,
    bs_alpha: float = 0.2,
    bs_threshold: float = 0.0,
    bs_direct_wt: float = 0.2,
    direct_map_mode: str = "exact",
    direct_topk: int = 3,
    direct_min_sim: float = 0.5,
    direct_vectors: Optional[str] = None,
    direct_fallback: bool = True,
    max_elems: int = 1 << 26,
    dense_elems: int = 1 << 25,
    log=print,
) -> Tuple[CSR, CSR, Optional[CSR]]:
    """Returns ``(Xf_Yf, Yf_Xf, direct_Xf_Yf)``."""
    X_Xf = trn_X_Xf.binarize()
    Y_Yf_b = Y_Yf.binarize()
    X_Y = trn_X_Y.binarize()

    log("counting co-occurrences...")
    X_Yf = spspmm(X_Y, Y_Yf_b, max_elems)  # (num_X, num_Yf): label features per point
    Y_Xf = spspmm(X_Y.transpose(), X_Xf, max_elems)  # (num_Y, num_Xf): point features per label

    # pair frequencies (over training point-label pairs) and marginal frequencies
    Yf_freq = torch.zeros(Y_Yf.ncols, dtype=X_Yf.values.dtype, device=X_Yf.device)
    Yf_freq.index_add_(0, X_Yf.indices, X_Yf.values)
    Xf_freq = torch.zeros(trn_X_Xf.ncols, dtype=Y_Xf.values.dtype, device=Y_Xf.device)
    Xf_freq.index_add_(0, Y_Xf.indices, Y_Xf.values)
    Xf_freq1 = torch.zeros_like(Xf_freq)
    Xf_freq1.index_add_(0, X_Xf.indices, X_Xf.values)
    Yf_freq1 = torch.zeros_like(Yf_freq)
    Yf_freq1.index_add_(0, Y_Yf_b.indices, Y_Yf_b.values)

    log("creating Yf_Xf...")
    Yf_Xf = _jaccard_topk(
        Y_Yf_b.transpose(), Y_Xf, Yf_freq, Xf_freq, Yf_freq1, Xf_freq1,
        bs_alpha, bs_count, bs_threshold, max_elems, dense_elems,
    )
    log("creating Xf_Yf...")
    Xf_Yf = _jaccard_topk(
        X_Xf.transpose(), X_Yf, Xf_freq, Yf_freq, Xf_freq1, Yf_freq1,
        bs_alpha, bs_count, bs_threshold, max_elems, dense_elems,
    )

    direct: Optional[CSR] = None
    if bs_direct_wt > 0:
        log("adding direct Xf-Yf matches...")
        direct = direct_map(
            Xf, Yf, bs_direct_wt, device=Xf_Yf.device, dtype=Xf_Yf.values.dtype,
            mode=direct_map_mode, topk=direct_topk, min_sim=direct_min_sim,
            vectors=direct_vectors, fallback_only=direct_fallback, max_elems=max_elems, dense_elems=dense_elems, log=log,
        )
        Xf_Yf = add_matrices(Xf_Yf, direct)

    Yf_Xf = remove_duplicates(Yf_Xf, Xf_Yf)
    return Xf_Yf.eliminate_zeros(), Yf_Xf.eliminate_zeros(), direct


def union_pattern(Xf_Yf: CSR, Yf_Xf: CSR) -> CSR:
    """``sparsity_pattern`` = ``Xf_Yf`` + ``Yf_Xf^T`` (the two supports are disjoint)."""
    return add_matrices(Xf_Yf, Yf_Xf.transpose())
