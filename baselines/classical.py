"""Three cheap classical baselines on GZ-NPM (key: `classical`).

1. centroid  : label = tf-idf centroid of its training points, cosine score.
2. knn       : k-NN label propagation over tf-idf cosine on training points.
3. bm25      : pure lexical BM25 of the point's tokens against the label's *text*
               (its "1_<token>" features). No training at all -> reaches unseen labels.

Writes Results/classical-<name>/score_mat.bin for each.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_num_threads(2)

from zestxml.csr import CSR  # noqa: E402
from zestxml.io import read_desc_file, read_text_smat, write_bin_smat  # noqa: E402

TOPK = 100


def to_scipy(m: CSR) -> sp.csr_matrix:
    return sp.csr_matrix(
        (m.values.numpy().astype(np.float64), m.indices.numpy(), m.indptr.numpy()), shape=m.shape
    )


def l2norm(m: sp.csr_matrix) -> sp.csr_matrix:
    n = np.sqrt(np.asarray(m.multiply(m).sum(1)).ravel())
    n[n == 0] = 1.0
    return sp.diags(1.0 / n) @ m


def save_topk(dense_chunks, n_labels, out_dir):
    """dense_chunks: list of (row_offset, ndarray n x n_labels). Keep top-TOPK per row."""
    rows, cols, vals = [], [], []
    for off, blk in dense_chunks:
        k = min(TOPK, blk.shape[1])
        idx = np.argpartition(-blk, k - 1, axis=1)[:, :k]
        v = np.take_along_axis(blk, idx, axis=1)
        keep = v > 0
        r = np.repeat(np.arange(blk.shape[0]) + off, keep.sum(1))
        rows.append(r)
        cols.append(idx[keep])
        vals.append(v[keep])
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals)
    n_rows = int(rows.max()) + 1 if rows.size else 0
    mat = CSR.from_coo(
        torch.as_tensor(rows).long(),
        torch.as_tensor(cols).long(),
        torch.as_tensor(vals).float(),
        (N_TST, n_labels),
    )
    os.makedirs(out_dir, exist_ok=True)
    write_bin_smat(mat, f"{out_dir}/score_mat.bin")
    print(f"wrote {out_dir}/score_mat.bin  nnz={mat.nnz} rows={n_rows}")


D = sys.argv[1] if len(sys.argv) > 1 else "GZXML-Datasets/GZ-NPM"
TAG = sys.argv[2] if len(sys.argv) > 2 else "classical"

t0 = time.time()
trn_X_Xf = l2norm(to_scipy(read_text_smat(f"{D}/trn_X_Xf.txt")))
tst_X_Xf = l2norm(to_scipy(read_text_smat(f"{D}/tst_X_Xf.txt")))
trn_X_Y = to_scipy(read_text_smat(f"{D}/trn_X_Y.txt"))
Y_Yf = to_scipy(read_text_smat(f"{D}/Y_Yf.txt"))
Xf = read_desc_file(f"{D}/Xf.txt")
Yf = read_desc_file(f"{D}/Yf.txt")
N_TST, N_LBL = tst_X_Xf.shape[0], trn_X_Y.shape[1]
print(f"loaded in {time.time()-t0:.1f}s  trn={trn_X_Xf.shape} tst={tst_X_Xf.shape} Y={Y_Yf.shape}")

CHUNK = 1500

# ------------------------------------------------------------------ 1. centroid
t = time.time()
C = l2norm(sp.csr_matrix(trn_X_Y.T @ trn_X_Xf))  # (labels x point-feats)
Ct = C.T.tocsc()
chunks = []
for i in range(0, N_TST, CHUNK):
    chunks.append((i, np.asarray((tst_X_Xf[i : i + CHUNK] @ Ct).todense())))
save_topk(chunks, N_LBL, f"Results/{TAG}-centroid")
print(f"centroid: {time.time()-t:.1f}s")
del chunks, C, Ct

# ------------------------------------------------------------------ 2. kNN
K = 25
t = time.time()
Xtrn_T = trn_X_Xf.T.tocsc()
chunks = []
for i in range(0, N_TST, CHUNK):
    S = np.asarray((tst_X_Xf[i : i + CHUNK] @ Xtrn_T).todense())
    kk = min(K, S.shape[1])
    idx = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
    v = np.take_along_axis(S, idx, axis=1)
    Ssp = sp.csr_matrix(
        (v.ravel(), idx.ravel(), np.arange(0, S.shape[0] * kk + 1, kk)), shape=S.shape
    )
    chunks.append((i, np.asarray((Ssp @ trn_X_Y).todense())))
save_topk(chunks, N_LBL, f"Results/{TAG}-knn{K}")
print(f"knn k={K}: {time.time()-t:.1f}s")
del chunks, Xtrn_T

# ------------------------------------------------------------------ 3. BM25
t = time.time()
# label documents: the "1_<token>" features of each label
lbl_tok = {}
for j, name in enumerate(Yf):
    if name.startswith("1_"):
        lbl_tok[j] = name[2:]
# point features -> tokens (unigram feature names, and each word of an n-gram name)
vocab = {}
for tok in lbl_tok.values():
    vocab.setdefault(tok, len(vocab))
pf_rows, pf_cols = [], []
for i, name in enumerate(Xf):
    for w in name.split():
        if w in vocab:
            pf_rows.append(i)
            pf_cols.append(vocab[w])
# (point-feature x token) incidence
PF2T = sp.csr_matrix(
    (np.ones(len(pf_rows)), (pf_rows, pf_cols)), shape=(len(Xf), len(vocab))
)
# (label x token) incidence, binary
ly, lt = [], []
Y_Yf_ind = Y_Yf.tocoo()
for r, c in zip(Y_Yf_ind.row, Y_Yf_ind.col):
    if c in lbl_tok:
        ly.append(r)
        lt.append(vocab[lbl_tok[c]])
Dm = sp.csr_matrix((np.ones(len(ly)), (ly, lt)), shape=(N_LBL, len(vocab)))
Dm.data[:] = 1.0
dl = np.asarray(Dm.sum(1)).ravel()
avgdl = dl[dl > 0].mean()
df = np.asarray((Dm > 0).sum(0)).ravel()
Nd = N_LBL
idf = np.log(1.0 + (Nd - df + 0.5) / (df + 0.5))
k1, b = 0.9, 0.4
denom = 1.0 + k1 * (1 - b + b * dl / avgdl)
Dw = sp.diags((k1 + 1) / denom) @ Dm @ sp.diags(idf)
Dwt = Dw.T.tocsc()

tst_bin = tst_X_Xf.copy()
tst_bin.data[:] = 1.0
Q = tst_bin @ PF2T  # (test x token) token counts
Qb = Q.copy()
Qb.data[:] = 1.0  # binary query
chunks = []
for i in range(0, N_TST, CHUNK):
    chunks.append((i, np.asarray((Qb[i : i + CHUNK] @ Dwt).todense())))
save_topk(chunks, N_LBL, f"Results/{TAG}-bm25")
print(f"bm25: {time.time()-t:.1f}s  vocab={len(vocab)} avgdl={avgdl:.2f}")
print(f"total {time.time()-t0:.1f}s")
