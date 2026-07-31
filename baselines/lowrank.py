"""ZestXML + a dense low-rank term:  score(x,y) = x^T (W_sparse + U V^T) y + b.

Reuses the mined sparsity pattern, the direct/knn map and both shortlists produced by a
plain ``run_torch.py`` run (``-res_dir <base>``), so the only difference against that
baseline is the extra low-rank term in the scorer.

    python3 baselines/lowrank.py -base Results/lowrank_base -res_dir Results/lowrank \
        -data GZXML-Datasets/GZ-NPM -rank 64 -lr_lowrank 0.02 -epochs 20

With ``-rank 0`` this is exactly the reference model (used as a sanity check that the
harness reproduces the baseline).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zestxml.csr import CSR, ranges  # noqa: E402
from zestxml.io import ensure_dir, read_bin_smat, read_text_smat, write_bin_smat  # noqa: E402
from zestxml.model import (  # noqa: E402
    BilinearClassifier,
    BilinearPattern,
    batch_points,
    direct_scores,
    pair_scores,
    pair_targets,
    score_transform,
    _loss,
)
from zestxml.pipeline import drop_unseen_labels, seen_labels_of  # noqa: E402


def log(msg=""):
    print(msg, flush=True)


class LowRankBilinear(BilinearClassifier):
    """``BilinearClassifier`` with an added rank-r term ``(U^T x) . (V^T y)``."""

    def __init__(self, pattern, n_xf, n_yf, rank=64, init_scale=0.05, seed=0, **kw):
        super().__init__(pattern, **kw)
        self.rank = rank
        g = torch.Generator(device="cpu").manual_seed(seed + 1234)
        dt, dev = self.weights.dtype, self.weights.device
        if rank > 0:
            self.U = (torch.randn(n_xf, rank, generator=g) * init_scale).to(dtype=dt, device=dev)
            self.V = (torch.randn(n_yf, rank, generator=g) * init_scale).to(dtype=dt, device=dev)
        else:
            self.U = torch.zeros(0, dtype=dt, device=dev)
            self.V = torch.zeros(0, dtype=dt, device=dev)
        self._Ysp = None  # cached sparse label matrix for the V projection

    def n_extra_params(self):
        return int(self.U.numel() + self.V.numel())

    def _label_proj(self, Y: CSR):
        """(num_Y x r) = Y @ V, cheap: one sparse mm over the whole label matrix."""
        if self._Ysp is None:
            self._Ysp = Y.to_torch_sparse()
        return torch.sparse.mm(self._Ysp, self.V)

    def _point_proj(self, X: CSR, points):
        start = X.indptr[points]
        cnt = X.indptr[points + 1] - start
        pos = ranges(start, cnt)
        seg = torch.repeat_interleave(torch.arange(points.numel(), device=X.device), cnt)
        src = self.U[X.indices[pos]] * X.values[pos].unsqueeze(1)
        return torch.zeros(points.numel(), self.rank, dtype=src.dtype, device=X.device).index_add(
            0, seg, src
        )

    def score_pairs(self, X, Y, pairs, points, norms=None):
        pos, owner = self._batch_pairs(pairs, points)
        labels = pairs.indices[pos]
        margin = pair_scores(self.weights, self.pattern, X, Y, points, owner, labels)
        if self.normalize:
            margin = margin / norms[pos]
        if self.rank > 0 and labels.numel():
            Px = self._point_proj(X, points)
            Py = self._label_proj(Y)
            margin = margin + (Px[owner] * Py[labels]).sum(1)
        return margin + self.bias, pos

    # ------------------------------------------------------------------- fit
    def fit_lowrank(self, X, Y, pairs, targets, epochs=20, lr=0.2, lr_lowrank=0.02,
                    reg_lowrank=1.0, batch_size=256, max_elems=1 << 24, seed=0, log=log):
        device = X.device
        gen = torch.Generator(device="cpu").manual_seed(seed)
        weight_per_pair = torch.full_like(targets, self.cost)
        weight_per_pair[targets > 0] *= self.pos_wt
        n_pairs = max(1, pairs.nnz)

        self.weights.requires_grad_(True)
        self.bias.requires_grad_(True)
        groups = [{"params": [self.weights, self.bias], "lr": lr}]
        if self.rank > 0:
            self.U.requires_grad_(True)
            self.V.requires_grad_(True)
            groups.append({"params": [self.U, self.V], "lr": lr_lowrank})
        opt = torch.optim.Adam(groups)
        base_lrs = [g["lr"] for g in opt.param_groups]

        for epoch in range(epochs):
            decay = 1.0 - epoch / max(1, epochs)
            for g, b in zip(opt.param_groups, base_lrs):
                g["lr"] = b * decay
            order = torch.randperm(X.nrows, generator=gen).to(device)
            total, seen = 0.0, 0
            t0 = time.time()
            for points in batch_points(X, self.pattern, order, max_elems, batch_size):
                margin, pos = self.score_pairs(X, Y, pairs, points)
                if margin.numel() == 0:
                    continue
                share = margin.numel() / n_pairs
                data_term = (weight_per_pair[pos] * _loss(margin, targets[pos], self.kind)).sum()
                reg = 0.5 * (self.weights.pow(2).sum() + self.bias.pow(2).sum()) * share
                if self.rank > 0:
                    reg = reg + 0.5 * reg_lowrank * (self.U.pow(2).sum() + self.V.pow(2).sum()) * share
                loss = (data_term + reg) / n_pairs
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                total += float(data_term.detach()) + float(reg.detach())
                seen += margin.numel()
            log("  epoch %d/%d objective %.4f over %d pairs (%.1f s)"
                % (epoch + 1, epochs, total, seen, time.time() - t0))

        for p in [self.weights, self.bias] + ([self.U, self.V] if self.rank > 0 else []):
            p.requires_grad_(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-data", default="GZXML-Datasets/GZ-NPM")
    ap.add_argument("-base", default="Results/lowrank_base")
    ap.add_argument("-res_dir", default="Results/lowrank")
    ap.add_argument("-rank", type=int, default=64)
    ap.add_argument("-lr", type=float, default=0.2)
    ap.add_argument("-lr_lowrank", type=float, default=0.02)
    ap.add_argument("-reg_lowrank", type=float, default=1.0)
    ap.add_argument("-init_scale", type=float, default=0.05)
    ap.add_argument("-epochs", type=int, default=20)
    ap.add_argument("-cost", type=float, default=5.0)
    ap.add_argument("-batch_size", type=int, default=256)
    ap.add_argument("-max_elems", type=int, default=1 << 26)
    ap.add_argument("-score_alpha", type=float, default=0.9)
    ap.add_argument("-num_thread", type=int, default=2)
    ap.add_argument("-seed", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(args.num_thread)

    D, base = args.data, args.base
    res = ensure_dir(args.res_dir)
    dtype, device = torch.float32, torch.device("cpu")

    log("loading data")
    trn_X_Xf = read_text_smat(D + "/trn_X_Xf.txt", device, dtype)
    tst_X_Xf = read_text_smat(D + "/tst_X_Xf.txt", device, dtype)
    Y_Yf_raw = read_text_smat(D + "/Y_Yf.txt", device, dtype)
    trn_X_Y = read_text_smat(D + "/trn_X_Y.txt", device, dtype)
    tst_X_Y = read_text_smat(D + "/tst_X_Y.txt", device, dtype)

    log("reusing mined pattern / shortlists from %s" % base)
    Xf_Yf = read_bin_smat(base + "/model/Xf_Yf.bin", device, dtype)
    Yf_Xf = read_bin_smat(base + "/model/Yf_Xf.bin", device, dtype)
    direct = read_bin_smat(base + "/model/direct_Xf_Yf.bin", device, dtype)
    trn_short = read_bin_smat(base + "/model/shortlist.bin", device, dtype)
    tst_short = read_bin_smat(base + "/shortlist.bin", device, dtype)

    seen = seen_labels_of(trn_X_Y)
    Y_trn = drop_unseen_labels(Y_Yf_raw, seen).unit_normalize_rows()
    Y_tst = Y_Yf_raw.unit_normalize_rows()
    trn_X_Xf = trn_X_Xf.unit_normalize_rows()
    tst_X_Xf = tst_X_Xf.unit_normalize_rows()

    pattern = BilinearPattern.from_parts(Xf_Yf, Yf_Xf)
    clf = LowRankBilinear(
        pattern, trn_X_Xf.ncols, Y_trn.ncols, rank=args.rank, init_scale=args.init_scale,
        seed=args.seed, cost=args.cost, normalize=False, device=device, dtype=dtype,
    )
    log("[STAT] sparse params : %d  low-rank params : %d  (rank %d)"
        % (pattern.size + 1, clf.n_extra_params(), args.rank))
    log("[STAT] train shortlist recall : %.2f%%" % trn_short.recall(trn_X_Y))
    log("[STAT] test  shortlist recall : %.2f%%" % tst_short.recall(tst_X_Y))

    targets = pair_targets(trn_short, trn_X_Y)
    t0 = time.time()
    clf.fit_lowrank(trn_X_Xf, Y_trn, trn_short, targets, epochs=args.epochs, lr=args.lr,
                    lr_lowrank=args.lr_lowrank, reg_lowrank=args.reg_lowrank,
                    batch_size=args.batch_size, max_elems=args.max_elems, seed=args.seed)
    log("training took %.1f s" % (time.time() - t0))

    log("predicting")
    clf._Ysp = None  # label matrix differs at test time (unseen labels keep their features)
    bilinear = clf.score_matrix(tst_X_Xf, Y_tst, tst_short, args.max_elems, transform=True)
    knn = direct_scores(direct, tst_X_Xf, Y_tst, tst_short, args.max_elems)
    a = args.score_alpha
    write_bin_smat(tst_short.with_values(a * bilinear + (1 - a) * knn), res + "/score_mat.bin")
    write_bin_smat(tst_short.with_values(bilinear), res + "/bilinear_score_mat.bin")
    log("wrote %s/score_mat.bin" % res)


if __name__ == "__main__":
    main()
