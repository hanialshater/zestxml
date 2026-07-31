"""A tiny CSR sparse matrix over torch tensors, plus the join primitives ZestXML needs.

The C++ implementation stores matrices as ``SMat``, a *column major* structure whose
``data[i]`` is the i-th column. Every matrix that ``SMat`` reads from a text file puts
one text line into one ``SMat`` column, so an ``SMat`` is exactly the CSR matrix of the
text file it was read from (``SMat.nc`` rows, ``SMat.nr`` columns). This module works in
that CSR view throughout: ``trn_X_Xf`` is (num points x num point-features), and so on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import torch
from torch import Tensor


def counts_to_indptr(counts: Tensor) -> Tensor:
    """[2, 0, 3] -> [0, 2, 2, 5]"""
    indptr = torch.zeros(counts.numel() + 1, dtype=torch.long, device=counts.device)
    torch.cumsum(counts, 0, out=indptr[1:])
    return indptr


def ranges(starts: Tensor, counts: Tensor) -> Tensor:
    """Concatenation of ``arange(s, s + c)`` for every (start, count) pair.

    This is the vectorised form of the "for each row, walk its non-zeros" loop that the
    C++ code writes by hand, and it is the workhorse of every routine below.
    """
    total = int(counts.sum())
    if total == 0:
        return torch.zeros(0, dtype=torch.long, device=starts.device)
    seg = torch.repeat_interleave(counts)  # segment id of each output element
    offsets = counts_to_indptr(counts)[:-1]
    return starts[seg] + (torch.arange(total, device=starts.device) - offsets[seg])


@dataclass
class CSR:
    """Row major sparse matrix. ``indices`` are column ids, sorted within each row."""

    indptr: Tensor  # int64, (nrows + 1,)
    indices: Tensor  # int64, (nnz,)
    values: Tensor  # float, (nnz,)
    shape: Tuple[int, int]

    # ---------------------------------------------------------------- basics
    @property
    def nrows(self) -> int:
        return self.shape[0]

    @property
    def ncols(self) -> int:
        return self.shape[1]

    @property
    def nnz(self) -> int:
        return int(self.indices.numel())

    @property
    def device(self) -> torch.device:
        return self.values.device

    def to(self, device=None, dtype=None) -> "CSR":
        return CSR(
            self.indptr.to(device) if device is not None else self.indptr,
            self.indices.to(device) if device is not None else self.indices,
            self.values.to(device=device, dtype=dtype),
            self.shape,
        )

    def clone(self) -> "CSR":
        return CSR(self.indptr.clone(), self.indices.clone(), self.values.clone(), self.shape)

    def row_counts(self) -> Tensor:
        return self.indptr[1:] - self.indptr[:-1]

    def row_ids(self) -> Tensor:
        """Row id of every non-zero."""
        return torch.repeat_interleave(self.row_counts())

    def keys(self, ncols: Optional[int] = None) -> Tensor:
        """``row * ncols + col`` for every non-zero -- a flat id used for set logic."""
        ncols = self.ncols if ncols is None else ncols
        return self.row_ids() * ncols + self.indices

    # ------------------------------------------------------------ construction
    @staticmethod
    def empty(shape, device=None, dtype=torch.float32) -> "CSR":
        return CSR(
            torch.zeros(shape[0] + 1, dtype=torch.long, device=device),
            torch.zeros(0, dtype=torch.long, device=device),
            torch.zeros(0, dtype=dtype, device=device),
            tuple(shape),
        )

    @staticmethod
    def from_coo(rows: Tensor, cols: Tensor, values: Tensor, shape, sort: bool = True) -> "CSR":
        """Build from triplets. Entries are ordered by (row, col); duplicates are kept."""
        if sort and rows.numel():
            order = torch.argsort(rows * shape[1] + cols)
            rows, cols, values = rows[order], cols[order], values[order]
        counts = torch.bincount(rows, minlength=shape[0])
        return CSR(counts_to_indptr(counts), cols, values, tuple(shape))

    @staticmethod
    def from_sorted_keys(keys: Tensor, values: Tensor, shape) -> "CSR":
        """Build from already ascending ``row * ncols + col`` keys."""
        rows = torch.div(keys, shape[1], rounding_mode="floor")
        counts = torch.bincount(rows, minlength=shape[0])
        return CSR(counts_to_indptr(counts), keys - rows * shape[1], values, tuple(shape))

    # -------------------------------------------------------------- operations
    def transpose(self) -> "CSR":
        rows = self.row_ids()
        order = torch.argsort(self.indices * self.nrows + rows)
        counts = torch.bincount(self.indices, minlength=self.ncols)
        return CSR(counts_to_indptr(counts), rows[order], self.values[order], (self.ncols, self.nrows))

    def binarize(self) -> "CSR":
        return CSR(self.indptr, self.indices, torch.ones_like(self.values), self.shape)

    def with_values(self, values: Tensor) -> "CSR":
        return CSR(self.indptr, self.indices, values, self.shape)

    def unit_normalize_rows(self) -> "CSR":
        """``SMat::unit_normalize_columns`` (L2), i.e. per text-row normalisation."""
        rows = self.row_ids()
        norm = torch.zeros(self.nrows, dtype=self.values.dtype, device=self.device)
        norm.index_add_(0, rows, self.values * self.values)
        norm = norm.sqrt()
        norm[norm == 0] = 1.0
        return self.with_values(self.values / norm[rows])

    def eliminate_zeros(self, eps: float = 1e-10) -> "CSR":
        keep = self.values.abs() > eps
        return self.subset(keep)

    def subset(self, keep: Tensor) -> "CSR":
        """Keep the non-zeros selected by a boolean mask, preserving order."""
        rows = self.row_ids()[keep]
        counts = torch.bincount(rows, minlength=self.nrows)
        return CSR(counts_to_indptr(counts), self.indices[keep], self.values[keep], self.shape)

    def threshold(self, th: float) -> "CSR":
        """``SMat::threshold`` -- drop entries with ``|v| <= th``."""
        return self.subset(self.values.abs() > th)

    def clear_rows(self, rows: Tensor) -> "CSR":
        """Empty out the given rows (``remove_test_labels``)."""
        drop = torch.zeros(self.nrows, dtype=torch.bool, device=self.device)
        drop[rows] = True
        return self.subset(~drop[self.row_ids()])

    def select_rows(self, rows: Tensor) -> "CSR":
        starts, counts = self.indptr[rows], self.indptr[rows + 1] - self.indptr[rows]
        pos = ranges(starts, counts)
        return CSR(counts_to_indptr(counts), self.indices[pos], self.values[pos], (rows.numel(), self.ncols))

    def sort_indices(self) -> "CSR":
        order = torch.argsort(self.keys())
        return CSR(self.indptr, self.indices[order], self.values[order], self.shape)

    def to_torch_sparse(self) -> Tensor:
        return torch.sparse_coo_tensor(
            torch.stack([self.row_ids(), self.indices]), self.values, self.shape
        ).coalesce()

    def to_dense(self) -> Tensor:
        out = torch.zeros(self.shape, dtype=self.values.dtype, device=self.device)
        out.view(-1).index_add_(0, self.keys(), self.values)
        return out

    def emultiply_nnz(self, other: "CSR") -> int:
        """Number of positions where both matrices are non-zero (used for recall)."""
        return int(torch.isin(self.keys(), other.keys()).sum())

    def recall(self, truth: "CSR") -> float:
        """``SMat::get_recall`` in percent."""
        return 100.0 * self.emultiply_nnz(truth) / max(1, truth.nnz)


# --------------------------------------------------------------------------- #
# products
# --------------------------------------------------------------------------- #
def row_costs(a: CSR, b: CSR) -> Tensor:
    """For each row of ``a``, the number of (a, b) non-zero pairs a product would touch."""
    return row_costs_from_counts(a, b.row_counts())


def row_costs_from_counts(a: CSR, counts: Tensor) -> Tensor:
    """``row_costs`` when only the right operand's row lengths are at hand."""
    per_nnz = counts[a.indices].to(torch.float64)
    cum = torch.zeros(a.nnz + 1, dtype=torch.float64, device=a.device)
    torch.cumsum(per_nnz, 0, out=cum[1:])
    return cum[a.indptr[1:]] - cum[a.indptr[:-1]]


def cost_chunks(costs: Tensor, max_elems: int) -> Iterator[Tuple[int, int]]:
    """Split rows into contiguous blocks whose summed cost stays under ``max_elems``."""
    n = costs.numel()
    lo = 0
    cum = torch.cumsum(costs.to(torch.float64), 0)
    while lo < n:
        base = float(cum[lo - 1]) if lo > 0 else 0.0
        bound = torch.tensor([base + max_elems], dtype=cum.dtype, device=cum.device)
        hi = int(torch.searchsorted(cum, bound, right=True))
        hi = min(max(hi, lo + 1), n)  # always make progress, even on a huge row
        yield lo, hi
        lo = hi


def expand_two_hop(a: CSR, b: CSR, rows: Tensor):
    """Non-zeros of ``a[rows] @ b`` before accumulation.

    Returns ``(seg, col, val)`` where ``seg`` indexes into ``rows``: the C++
    ``prod_helper`` loop, vectorised and left un-summed.
    """
    a_start = a.indptr[rows]
    a_cnt = a.indptr[rows + 1] - a_start
    a_pos = ranges(a_start, a_cnt)
    mid = a.indices[a_pos]
    a_val = a.values[a_pos]
    seg = torch.repeat_interleave(torch.arange(rows.numel(), device=a.device), a_cnt)

    b_start = b.indptr[mid]
    b_cnt = b.indptr[mid + 1] - b_start
    b_pos = ranges(b_start, b_cnt)
    return (
        torch.repeat_interleave(seg, b_cnt),
        b.indices[b_pos],
        torch.repeat_interleave(a_val, b_cnt) * b.values[b_pos],
    )


def prod_dense_rows(a: CSR, b: CSR, rows: Tensor, out: Optional[Tensor] = None) -> Tensor:
    """Dense ``a[rows] @ b``. Caller keeps ``rows`` small enough for the dense block."""
    if out is None:
        out = torch.zeros(rows.numel(), b.ncols, dtype=a.values.dtype, device=a.device)
    else:
        out.zero_()
    seg, col, val = expand_two_hop(a, b, rows)
    out.view(-1).index_add_(0, seg * b.ncols + col, val)
    return out


def spspmm(a: CSR, b: CSR, max_elems: int = 1 << 26) -> CSR:
    """Sparse ``a @ b`` as CSR, accumulated chunk by chunk to bound peak memory."""
    assert a.ncols == b.nrows, f"shape mismatch: {a.shape} @ {b.shape}"
    costs = row_costs(a, b)
    keys_out, vals_out = [], []
    for lo, hi in cost_chunks(costs, max_elems):
        rows = torch.arange(lo, hi, device=a.device)
        seg, col, val = expand_two_hop(a, b, rows)
        if seg.numel() == 0:
            continue
        keys = (seg + lo) * b.ncols + col
        uniq, inv = torch.unique(keys, return_inverse=True)
        summed = torch.zeros(uniq.numel(), dtype=val.dtype, device=val.device).index_add_(0, inv, val)
        keys_out.append(uniq)
        vals_out.append(summed)
    if not keys_out:
        return CSR.empty((a.nrows, b.ncols), device=a.device, dtype=a.values.dtype)
    return CSR.from_sorted_keys(torch.cat(keys_out), torch.cat(vals_out), (a.nrows, b.ncols))


def topk_per_row_dense(block: Tensor, k: int, extra_above: Optional[float] = None) -> Tensor:
    """Boolean mask of the top-``k`` positive entries of each row of a dense block.

    Mirrors ``prod_for_jaccard``'s retention rule: keep the ``k`` largest non-zeros, plus
    every entry above ``extra_above`` (the ``good_th`` escape hatch in the C++ code).
    """
    keep = torch.zeros_like(block, dtype=torch.bool)
    kk = min(k, block.shape[1])
    if kk > 0:
        _, idx = torch.topk(block, kk, dim=1)
        keep.scatter_(1, idx, True)
    keep &= block > 0
    if extra_above is not None:
        keep |= block > extra_above
    return keep
