"""The ZestXML model: shortlisting, the sparse bilinear scorer, and its training loop.

The model is a bilinear form ``score(x, y) = x^T W y + b`` where ``W`` (num_Xf x num_Yf)
is constrained to the sparsity pattern mined in :mod:`zestxml.pattern`. Because ``W`` is
sparse and fixed in support, its free parameters are just one vector ``w`` over the
pattern's non-zeros -- exactly the vector the C++ code learns after flattening every
(point, label) pair into the "linear form" of ``bilinear_to_linear_form``.

Rather than materialising that linear form (an ``|pairs| x |w|`` sparse matrix), the
scorer here contracts the three sparse operands directly:

    score(x, y) = sum_{yf in y} y[yf] * ( sum_{(xf, yf) in S, xf in x} w[xf, yf] * x[xf] )

The inner sum is accumulated per (point, label-feature) with a sorted-key join, so peak
memory depends on the batch, never on num_Yf or num_Y. Everything is differentiable in
``w``, so the primal objective is optimised directly instead of through liblinear's dual
coordinate descent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
from torch import Tensor

from .csr import (
    CSR,
    bounded_chunks,
    cost_chunks,
    counts_to_indptr,
    prod_dense_rows,
    ranges,
    row_costs,
    row_costs_from_counts,
)

L2R_L2LOSS_SVC = 0  # squared hinge, liblinear's L2R_L2LOSS_SVC_DUAL
L2R_LR = 1  # logistic, liblinear's L2R_LR_DUAL


# --------------------------------------------------------------------------- #
# the pattern, viewed from the point-feature side
# --------------------------------------------------------------------------- #
@dataclass
class BilinearPattern:
    """Pattern entries grouped by ``xf``, each carrying its slot in the weight vector."""

    indptr: Tensor  # (num_Xf + 1,)
    yf: Tensor  # (P,)
    pid: Tensor  # (P,) index into the weight vector
    n_xf: int
    n_yf: int

    @property
    def size(self) -> int:
        return int(self.yf.numel())

    def row_counts(self) -> Tensor:
        return self.indptr[1:] - self.indptr[:-1]

    def to(self, device) -> "BilinearPattern":
        return BilinearPattern(
            self.indptr.to(device), self.yf.to(device), self.pid.to(device), self.n_xf, self.n_yf
        )

    def as_csr(self, values: Tensor) -> CSR:
        return CSR(self.indptr, self.yf, values[self.pid], (self.n_xf, self.n_yf))

    @staticmethod
    def from_parts(Xf_Yf: CSR, Yf_Xf: CSR) -> "BilinearPattern":
        """Weight vector layout, matching ``expand_linear_form_mat``:
        ``Xf_Yf``'s non-zeros in order, then ``Yf_Xf``'s."""
        device = Xf_Yf.device
        xf = torch.cat([Xf_Yf.row_ids(), Yf_Xf.indices])
        yf = torch.cat([Xf_Yf.indices, Yf_Xf.row_ids()])
        pid = torch.arange(Xf_Yf.nnz + Yf_Xf.nnz, device=device)

        order = torch.argsort(xf * Xf_Yf.ncols + yf)
        xf, yf, pid = xf[order], yf[order], pid[order]
        counts = torch.bincount(xf, minlength=Xf_Yf.nrows)
        return BilinearPattern(counts_to_indptr(counts), yf, pid, Xf_Yf.nrows, Xf_Yf.ncols)

    @staticmethod
    def from_csr(mat: CSR) -> "BilinearPattern":
        """A pattern whose entries all share weight slot 0 (used for the direct/knn map)."""
        return BilinearPattern(
            mat.indptr, mat.indices, torch.zeros(mat.nnz, dtype=torch.long, device=mat.device),
            mat.nrows, mat.ncols,
        )


# --------------------------------------------------------------------------- #
# the scorer
# --------------------------------------------------------------------------- #
def pair_scores(
    weights: Tensor,
    pattern: BilinearPattern,
    X: CSR,
    Y: CSR,
    points: Tensor,
    pair_owner: Tensor,
    pair_label: Tensor,
    square: bool = False,
) -> Tensor:
    """Bilinear score of every (point, label) pair in a batch.

    ``points`` are global point ids; ``pair_owner[j]`` indexes into ``points`` and
    ``pair_label[j]`` is the label of pair ``j``. With ``square=True`` the weights are
    ignored and ``sum_p (x[xf_p] * y[yf_p])^2`` is returned instead, which is the squared
    norm of the pair's linear-form feature vector.
    """
    n_pairs = pair_label.numel()
    out = torch.zeros(n_pairs, dtype=X.values.dtype, device=X.device)
    if n_pairs == 0 or points.numel() == 0:
        return out

    # label side: the (point, yf) cells this batch actually needs
    l_start = Y.indptr[pair_label]
    l_cnt = Y.indptr[pair_label + 1] - l_start
    l_pos = ranges(l_start, l_cnt)
    l_pair = torch.repeat_interleave(torch.arange(n_pairs, device=X.device), l_cnt)
    l_yf = Y.indices[l_pos]
    l_val = Y.values[l_pos]
    if square:
        l_val = l_val * l_val

    cell_key = pair_owner[l_pair] * pattern.n_yf + l_yf
    cells, cell_of = torch.unique(cell_key, return_inverse=True)
    if cells.numel() == 0:
        return out

    # point side: walk each point's features into the pattern rows they activate
    x_start = X.indptr[points]
    x_cnt = X.indptr[points + 1] - x_start
    x_pos = ranges(x_start, x_cnt)
    x_owner = torch.repeat_interleave(torch.arange(points.numel(), device=X.device), x_cnt)
    xf = X.indices[x_pos]
    x_val = X.values[x_pos]

    p_start = pattern.indptr[xf]
    p_cnt = pattern.indptr[xf + 1] - p_start
    p_pos = ranges(p_start, p_cnt)
    contrib = torch.repeat_interleave(x_val, p_cnt)
    if square:
        contrib = contrib * contrib
    else:
        contrib = contrib * weights[pattern.pid[p_pos]]

    key = torch.repeat_interleave(x_owner, p_cnt) * pattern.n_yf + pattern.yf[p_pos]
    slot = torch.searchsorted(cells, key).clamp(max=cells.numel() - 1)
    hit = cells[slot] == key

    acc = torch.zeros(cells.numel(), dtype=out.dtype, device=out.device).index_add(
        0, slot[hit], contrib[hit]
    )
    return out.index_add(0, l_pair, acc[cell_of] * l_val)


def batch_points(X: CSR, pattern: BilinearPattern, order: Tensor, max_elems: int, batch_size=None):
    """Yield point batches, capped by expansion cost and optionally by point count."""
    costs = row_costs_from_counts(X, pattern.row_counts())[order]
    for lo, hi in bounded_chunks(costs, max_elems, batch_size):
        yield order[lo:hi]


# --------------------------------------------------------------------------- #
# shortlisting
# --------------------------------------------------------------------------- #
def get_shortlist(
    X_Xf: CSR,
    Y_Yf: CSR,
    sparsity_pattern: CSR,
    shortyK: int,
    max_elems: int = 1 << 26,
    dense_elems: int = 1 << 24,
    log=print,
) -> CSR:
    """Top-``shortyK`` labels per point under ``X_Xf @ sparsity_pattern @ Y_Yf^T``.

    ``get_approx_shortlist`` in ``Source/zestxml.cpp`` walks a per-label-feature inverted
    index with a shrinking threshold to approximate this top-k; here the same quantity is
    computed exactly in chunks, which is both simpler to vectorise and strictly better
    for shortlist recall (``get_exact_shortlist`` is the C++ equivalent).
    """
    device = X_Xf.device
    n_points, n_labels = X_Xf.nrows, Y_Yf.nrows
    k = min(shortyK, n_labels)
    Y_sparse = Y_Yf.to_torch_sparse()

    costs = row_costs(X_Xf, sparsity_pattern)
    max_rows = max(1, dense_elems // max(1, max(sparsity_pattern.ncols, n_labels)))

    counts = torch.zeros(n_points, dtype=torch.long, device=device)
    idx_out, val_out = [], []
    done = 0
    for lo, hi in cost_chunks(costs, max_elems):
        for sub in range(lo, hi, max_rows):
            rows = torch.arange(sub, min(sub + max_rows, hi), device=device)
            proj = prod_dense_rows(X_Xf, sparsity_pattern, rows)  # (b, num_Yf)
            scores = torch.sparse.mm(Y_sparse, proj.t().contiguous()).t()  # (b, num_Y)
            del proj

            top_val, top_idx = torch.topk(scores, k, dim=1)
            del scores
            keep = top_val > 0  # a zero score means no shared label feature at all
            row_counts = keep.sum(1)
            # order each row's shortlist by label id, as the C++ writer does
            sort_key = torch.where(keep, top_idx, torch.full_like(top_idx, n_labels))
            sort_key, perm = torch.sort(sort_key, dim=1)
            top_val = torch.gather(top_val, 1, perm)
            valid = torch.arange(k, device=device)[None, :] < row_counts[:, None]

            counts[rows] = row_counts
            idx_out.append(sort_key[valid])
            val_out.append(top_val[valid])
            done += rows.numel()
            if done % 100000 < rows.numel():
                log(f"  shortlisted {done}/{n_points} points")

    return CSR(
        counts_to_indptr(counts),
        torch.cat(idx_out) if idx_out else torch.zeros(0, dtype=torch.long, device=device),
        torch.cat(val_out) if val_out else torch.zeros(0, dtype=X_Xf.values.dtype, device=device),
        (n_points, n_labels),
    )


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def score_transform(scores: Tensor, kind: int) -> Tensor:
    """``get_new_score`` with ``logp=false``: map a margin to a positive score."""
    if kind == L2R_L2LOSS_SVC:
        return torch.exp(-torch.clamp(1.0 - scores, min=0.0) ** 2)
    if kind == L2R_LR:
        return torch.sigmoid(scores)
    return scores.abs()


def _loss(margin: Tensor, target: Tensor, kind: int) -> Tensor:
    z = target * margin
    if kind == L2R_L2LOSS_SVC:
        return torch.clamp(1.0 - z, min=0.0) ** 2
    if kind == L2R_LR:
        return torch.nn.functional.softplus(-z)
    raise ValueError(f"unsupported classifier kind: {kind}")


class BilinearClassifier:
    """Learns ``w`` (plus a bias) for the shortlisted (point, label) pairs.

    Objective, matching liblinear's primal for the chosen loss::

        0.5 * ||w||^2 + sum_i C_i * loss(y_i * w . phi_i)

    with ``C_i = cost`` (times ``pos_wt`` for positives). Minimised with Adam over
    mini-batches of points instead of dual coordinate descent, which keeps every step a
    handful of sparse gather/scatter kernels and runs on GPU.
    """

    def __init__(
        self,
        pattern: BilinearPattern,
        kind: int = L2R_L2LOSS_SVC,
        cost: float = 1.0,
        pos_wt: float = 1.0,
        normalize: bool = False,
        device=None,
        dtype=torch.float32,
    ):
        self.pattern = pattern
        self.kind = kind
        self.cost = cost
        self.pos_wt = pos_wt
        self.normalize = normalize
        self.device = device
        self.weights = torch.zeros(pattern.size, dtype=dtype, device=device)
        self.bias = torch.zeros(1, dtype=dtype, device=device)

    # ------------------------------------------------------------------ scoring
    def _batch_pairs(self, pairs: CSR, points: Tensor) -> Tuple[Tensor, Tensor]:
        """Global pair ids of ``points``' pairs, and which point each belongs to."""
        starts = pairs.indptr[points]
        counts = pairs.indptr[points + 1] - starts
        pos = ranges(starts, counts)
        owner = torch.repeat_interleave(torch.arange(points.numel(), device=pairs.device), counts)
        return pos, owner

    def score_pairs(
        self, X: CSR, Y: CSR, pairs: CSR, points: Tensor, norms: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """Margins for the pairs of ``points`` plus their global pair ids."""
        pos, owner = self._batch_pairs(pairs, points)
        margin = pair_scores(self.weights, self.pattern, X, Y, points, owner, pairs.indices[pos])
        if self.normalize:
            margin = margin / norms[pos]
        return margin + self.bias, pos

    def score_matrix(
        self,
        X: CSR,
        Y: CSR,
        pairs: CSR,
        max_elems: int,
        transform: bool = False,
        norms: Optional[Tensor] = None,
    ) -> Tensor:
        """Score every pair of ``pairs``, batch by batch, aligned with ``pairs.indices``."""
        if self.normalize and norms is None:
            norms = self.pair_norms(X, Y, pairs, max_elems)
        out = torch.zeros(pairs.nnz, dtype=X.values.dtype, device=X.device)
        order = torch.arange(X.nrows, device=X.device)
        with torch.no_grad():
            for points in batch_points(X, self.pattern, order, max_elems):
                margin, pos = self.score_pairs(X, Y, pairs, points, norms)
                out[pos] = score_transform(margin, self.kind) if transform else margin
        return out

    def pair_norms(self, X: CSR, Y: CSR, pairs: CSR, max_elems: int, log=print) -> Tensor:
        """L2 norms of the linear-form feature vectors (``bilinear_normalize``)."""
        norms = torch.zeros(pairs.nnz, dtype=X.values.dtype, device=X.device)
        order = torch.arange(X.nrows, device=X.device)
        with torch.no_grad():
            for points in batch_points(X, self.pattern, order, max_elems):
                pos, owner = self._batch_pairs(pairs, points)
                sq = pair_scores(
                    self.weights, self.pattern, X, Y, points, owner, pairs.indices[pos], square=True
                )
                norms[pos] = sq.clamp_min(0).sqrt()
        norms[norms == 0] = 1.0
        return norms

    # ----------------------------------------------------------------- training
    def fit(
        self,
        X: CSR,
        Y: CSR,
        pairs: CSR,
        targets: Tensor,
        epochs: int = 20,
        lr: float = 0.2,
        batch_size: int = 256,
        max_elems: int = 1 << 24,
        seed: int = 0,
        log=print,
    ) -> None:
        device = X.device
        generator = torch.Generator(device="cpu").manual_seed(seed)

        norms = self.pair_norms(X, Y, pairs, max_elems, log=log) if self.normalize else None
        weight_per_pair = torch.full_like(targets, self.cost)
        weight_per_pair[targets > 0] *= self.pos_wt
        n_pairs = max(1, pairs.nnz)

        self.weights.requires_grad_(True)
        self.bias.requires_grad_(True)
        opt = torch.optim.Adam([self.weights, self.bias], lr=lr)

        for epoch in range(epochs):
            for group in opt.param_groups:  # linear decay, so late epochs settle
                group["lr"] = lr * (1.0 - epoch / max(1, epochs))
            order = torch.randperm(X.nrows, generator=generator).to(device)
            total, seen = 0.0, 0
            for points in batch_points(X, self.pattern, order, max_elems, batch_size):
                margin, pos = self.score_pairs(X, Y, pairs, points, norms)
                if margin.numel() == 0:
                    continue
                data_term = (weight_per_pair[pos] * _loss(margin, targets[pos], self.kind)).sum()
                reg = 0.5 * (self.weights.pow(2).sum() + self.bias.pow(2).sum()) * (margin.numel() / n_pairs)
                loss = (data_term + reg) / n_pairs

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                total += float(data_term.detach()) + float(reg.detach())
                seen += margin.numel()
            log(f"  epoch {epoch + 1}/{epochs} objective {total:.4f} over {seen} pairs")

        self.weights.requires_grad_(False)
        self.bias.requires_grad_(False)

    def training_report(self, X: CSR, Y: CSR, pairs: CSR, targets: Tensor, max_elems: int, log=print) -> None:
        """The "acc / pos acc / neg acc" line ``learn_bilinear_classifier`` prints."""
        pos_acc, neg_acc, n_pos, n_neg = 0.0, 0.0, 0, 0
        order = torch.arange(X.nrows, device=X.device)
        norms = self.pair_norms(X, Y, pairs, max_elems) if self.normalize else None
        with torch.no_grad():
            for points in batch_points(X, self.pattern, order, max_elems):
                margin, pos = self.score_pairs(X, Y, pairs, points, norms)
                scores = score_transform(margin, self.kind)
                is_pos = targets[pos] > 0
                pos_acc += float(scores[is_pos].sum())
                neg_acc += float((1 - scores[~is_pos]).sum())
                n_pos += int(is_pos.sum())
                n_neg += int((~is_pos).sum())
        n = max(1, n_pos + n_neg)
        log(
            "acc : %.2f%% pos acc : %.2f%% neg acc : %.2f%%"
            % (100.0 * (pos_acc + neg_acc) / n, 100.0 * pos_acc / max(1, n_pos), 100.0 * neg_acc / max(1, n_neg))
        )


def direct_scores(direct: CSR, X: CSR, Y: CSR, pairs: CSR, max_elems: int) -> Tensor:
    """The knn-style score: how well a label's own tokens match the point's tokens.

    ``predict`` in the C++ builds this from ``direct_Xf_Yf`` with all values set to 1.
    """
    # the C++ sets every direct weight to 1; scaling by the max reproduces that for an
    # exact map and keeps fuzzy links proportional to their similarity
    scale = float(direct.values.abs().max()) if direct.nnz else 1.0
    pattern = BilinearPattern.from_csr(direct.with_values(direct.values / max(scale, 1e-12)))
    clf = BilinearClassifier(pattern, device=X.device, dtype=X.values.dtype)
    clf.weights = torch.ones(1, dtype=X.values.dtype, device=X.device)
    clf.bias = torch.zeros(1, dtype=X.values.dtype, device=X.device)
    return clf.score_matrix(X, Y, pairs, max_elems)


def pair_targets(pairs: CSR, truth: CSR) -> Tensor:
    """+1 for shortlisted pairs that are true positives, -1 otherwise."""
    hit = torch.isin(pairs.keys(), truth.keys())
    return torch.where(hit, 1.0, -1.0).to(pairs.values.dtype)
