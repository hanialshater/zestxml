"""One-vs-all linear classifiers (squared hinge + L2) -- the classic XMC baseline.

Dense weight matrix (n_point_features x n_labels) trained full-batch with Adam.
Gradients are written by hand so nothing but the score block is materialised.

This model has no notion of label features, so a label with no training positive
gets an all-zero (never-updated except by L2) column: it can never be predicted.
That is the point of the experiment.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from zestxml.csr import CSR  # noqa: E402
from zestxml.io import read_text_smat, write_bin_smat, ensure_dir  # noqa: E402

torch.set_num_threads(2)


def add_bias(X: CSR) -> torch.Tensor:
    """Sparse (n x d+1) COO tensor with a constant 1 appended to every row."""
    rows = X.row_ids()
    n, d = X.shape
    idx = torch.stack([
        torch.cat([rows, torch.arange(n)]),
        torch.cat([X.indices, torch.full((n,), d, dtype=torch.long)]),
    ])
    val = torch.cat([X.values, torch.ones(n)])
    return torch.sparse_coo_tensor(idx, val, (n, d + 1)).coalesce().to_sparse_csr()


def dense_targets(Y: CSR) -> torch.Tensor:
    T = -torch.ones(Y.shape, dtype=torch.float32)
    T[Y.row_ids(), Y.indices] = 1.0
    return T


def transpose_csr(Xs):
    c = Xs.to_sparse_coo()
    i = c.indices()
    return torch.sparse_coo_tensor(torch.stack([i[1], i[0]]), c.values(),
                                   (Xs.shape[1], Xs.shape[0])).coalesce().to_sparse_csr()


def train(Xs, T, lam, epochs, lr, pos_wt, log=True):
    n, d = Xs.shape
    L = T.shape[1]
    XsT = transpose_csr(Xs)
    W = torch.zeros(d, L)
    opt = torch.optim.Adam([W], lr=lr)
    wt = torch.where(T > 0, torch.tensor(pos_wt), torch.tensor(1.0))
    for ep in range(epochs):
        S = Xs @ W
        m = (1.0 - T * S).clamp_(min=0)
        loss = (wt * m * m).sum() / n
        G = (-2.0 / n) * wt * T * m
        gW = XsT @ G
        gW += 2.0 * lam * W
        W.grad = gW
        opt.step()
        if log and (ep % 5 == 0 or ep == epochs - 1):
            print(f"  ep {ep:3d} loss {loss.item():.5f} +reg {lam * (W * W).sum().item():.5f}",
                  flush=True)
    return W


def topk_scores(Xs, W, k=100):
    """Sigmoid-squashed top-k scores per row (positive, so they beat the implicit 0)."""
    n = Xs.shape[0]
    L = W.shape[1]
    S = Xs @ W
    kk = min(k, L)
    v, i = torch.topk(S, kk, dim=1)
    keys = torch.arange(n).repeat_interleave(kk) * L + i.reshape(-1)
    vals = torch.sigmoid(v.reshape(-1))
    order = torch.argsort(keys)
    return CSR.from_sorted_keys(keys[order], vals[order], (n, L))


def p_at_1(Xs, W, Ytrue: CSR):
    S = Xs @ W
    top = S.argmax(1)
    hit = torch.zeros(Ytrue.shape, dtype=torch.bool)
    hit[Ytrue.row_ids(), Ytrue.indices] = True
    keep = Ytrue.row_counts() > 0
    return 100.0 * hit[torch.arange(Ytrue.shape[0]), top][keep].float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-data", default="GZXML-Datasets/GZ-NPM")
    ap.add_argument("-res_dir", default="Results/ova_linear")
    ap.add_argument("-lam", type=float, default=1e-4)
    ap.add_argument("-epochs", type=int, default=40)
    ap.add_argument("-lr", type=float, default=0.05)
    ap.add_argument("-pos_wt", type=float, default=1.0)
    ap.add_argument("-topk", type=int, default=100)
    ap.add_argument("-val", action="store_true", help="hold out 10%% of train, report val P@1, no test output")
    args = ap.parse_args()

    t0 = time.time()
    trn_X = read_text_smat(f"{args.data}/trn_X_Xf.txt")
    trn_Y = read_text_smat(f"{args.data}/trn_X_Y.txt")
    print(f"train {trn_X.shape} nnz {trn_X.nnz}, labels {trn_Y.shape[1]}", flush=True)

    if args.val:
        g = torch.Generator().manual_seed(0)
        perm = torch.randperm(trn_X.shape[0], generator=g)
        ntr = int(0.9 * perm.numel())
        tr, va = perm[:ntr].sort().values, perm[ntr:].sort().values
        Xtr, Ytr = trn_X.select_rows(tr), trn_Y.select_rows(tr)
        Xva, Yva = trn_X.select_rows(va), trn_Y.select_rows(va)
        Xs = add_bias(Xtr)
        W = train(Xs, dense_targets(Ytr), args.lam, args.epochs, args.lr, args.pos_wt)
        print(f"VAL lam={args.lam} epochs={args.epochs} lr={args.lr} pos_wt={args.pos_wt} "
              f"P@1={p_at_1(add_bias(Xva), W, Yva):.2f}  ({time.time() - t0:.0f}s)", flush=True)
        return

    Xs = add_bias(trn_X)
    W = train(Xs, dense_targets(trn_Y), args.lam, args.epochs, args.lr, args.pos_wt)

    # verify the unseen-label columns really are dead
    lab_freq = torch.zeros(trn_Y.shape[1]).index_add_(0, trn_Y.indices, torch.ones(trn_Y.nnz))
    unseen = lab_freq == 0
    print(f"unseen label columns: {int(unseen.sum())}, max |w| over them = "
          f"{W[:, unseen].abs().max().item():.3e}, max |w| over seen = "
          f"{W[:, ~unseen].abs().max().item():.3e}", flush=True)

    tst_X = read_text_smat(f"{args.data}/tst_X_Xf.txt")
    smat = topk_scores(add_bias(tst_X), W, args.topk)
    ensure_dir(args.res_dir)
    write_bin_smat(smat, f"{args.res_dir}/score_mat.bin")
    print(f"wrote {args.res_dir}/score_mat.bin nnz={smat.nnz} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
