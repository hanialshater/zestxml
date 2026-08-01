"""Untuned 1:1 hybrid of the knn and bm25 score matrices (row-max normalised),
plus candidate-set recall@100 for each of the three classical baselines."""
from __future__ import annotations

import os
import sys

import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
torch.set_num_threads(2)
from zestxml.csr import CSR  # noqa: E402
from zestxml.io import read_bin_smat, read_text_smat, write_bin_smat  # noqa: E402

D = sys.argv[1]
TAG = sys.argv[2]


def to_scipy(m):
    return sp.csr_matrix((m.values.numpy().astype(np.float64), m.indices.numpy(), m.indptr.numpy()), shape=m.shape)


def rownorm(m):
    mx = np.asarray(m.max(1).todense()).ravel()
    mx[mx <= 0] = 1.0
    return sp.diags(1.0 / mx) @ m


tst = to_scipy(read_text_smat(f"{D}/tst_X_Y.txt"))
tst.data[:] = 1.0
npos = np.asarray(tst.sum(1)).ravel()
for name in ["centroid", "knn25", "bm25"]:
    S = to_scipy(read_bin_smat(f"Results/{TAG}-{name}/score_mat.bin"))
    Sb = S.copy()
    Sb.data[:] = 1.0
    hit = np.asarray(Sb.multiply(tst).sum(1)).ravel()
    ok = npos > 0
    print(f"{name:9s} recall@100 = {100.0*(hit[ok]/npos[ok]).mean():.2f}")

A = rownorm(to_scipy(read_bin_smat(f"Results/{TAG}-knn25/score_mat.bin")))
B = rownorm(to_scipy(read_bin_smat(f"Results/{TAG}-bm25/score_mat.bin")))
H = (A + B).tocoo()
mat = CSR.from_coo(
    torch.as_tensor(H.row).long(), torch.as_tensor(H.col).long(), torch.as_tensor(H.data).float(), H.shape
)
os.makedirs(f"Results/{TAG}-hybrid", exist_ok=True)
write_bin_smat(mat, f"Results/{TAG}-hybrid/score_mat.bin")
print("hybrid nnz", mat.nnz)
