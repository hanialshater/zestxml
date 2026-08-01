"""Readers/writers for the file formats used by the C++ implementation.

Both the text format (``<num_row> <num_col>`` header followed by ``idx:val`` lines) and
the binary ``.bin`` dumps are byte-compatible with the C++ ``SMat``, so models and score
matrices written by either implementation can be read by the other. That compatibility is
what let this port be validated against the reference; the C++ itself no longer lives here.
"""

from __future__ import annotations

import os
from typing import List

import numpy as np
import torch

from .csr import CSR


# --------------------------------------------------------------------------- #
# text sparse matrices
# --------------------------------------------------------------------------- #
def read_text_smat(path: str, device=None, dtype=torch.float32) -> CSR:
    """Read the plain text sparse matrix format described in the README."""
    indices: List[np.ndarray] = []
    values: List[np.ndarray] = []
    with open(path) as f:
        header = f.readline().split()
        nrows, ncols = int(header[0]), int(header[1])
        counts = np.zeros(nrows, dtype=np.int64)
        for i, line in enumerate(f):
            if i >= nrows:
                break
            if not line.strip():
                continue
            flat = np.array(line.replace(":", " ").split(), dtype=np.float64)
            indices.append(flat[0::2].astype(np.int64))
            values.append(flat[1::2])
            counts[i] = flat.size // 2

    idx = np.concatenate(indices) if indices else np.zeros(0, dtype=np.int64)
    val = np.concatenate(values) if values else np.zeros(0)
    mat = CSR(
        indptr=torch.as_tensor(np.concatenate([[0], counts.cumsum()])).to(torch.long),
        indices=torch.as_tensor(idx).to(torch.long),
        values=torch.as_tensor(val).to(dtype),
        shape=(nrows, ncols),
    )
    # the C++ reader sorts every row by column id; do the same so both agree bit for bit
    return mat.sort_indices().to(device)


def write_text_smat(mat: CSR, path: str) -> None:
    indptr = mat.indptr.cpu().numpy()
    indices = mat.indices.cpu().numpy()
    values = mat.values.cpu().numpy()
    with open(path, "w") as f:
        f.write("%d %d\n" % (mat.nrows, mat.ncols))
        for i in range(mat.nrows):
            lo, hi = indptr[i], indptr[i + 1]
            f.write(" ".join("%d:%.5f" % (j, v) for j, v in zip(indices[lo:hi], values[lo:hi])) + "\n")


def read_desc_file(path: str) -> List[str]:
    """Feature description files (``Xf.txt`` / ``Yf.txt``).

    The C++ ``read_desc_file`` reads lines until eof and then unconditionally
    drops the last one; splitting on newline and dropping the final element reproduces
    that exactly, trailing newline or not.
    """
    with open(path) as f:
        return f.read().split("\n")[:-1]


# --------------------------------------------------------------------------- #
# binary sparse matrices (SMat::writeBin / readBin)
# --------------------------------------------------------------------------- #
# layout: int32 num_rows, int32 num_cols, int64 num_rows, int32 row sizes,
#         then (int32 col, float32 val) pairs row by row.
_PAIR = np.dtype([("idx", np.int32), ("val", np.float32)])


def read_bin_smat(path: str, device=None, dtype=torch.float32) -> CSR:
    buf = np.fromfile(path, dtype=np.uint8)
    nrows, ncols = np.frombuffer(buf, dtype=np.int32, count=2, offset=0)
    off = 8
    n_sizes = int(np.frombuffer(buf, dtype=np.int64, count=1, offset=off)[0])
    off += 8
    sizes = np.frombuffer(buf, dtype=np.int32, count=n_sizes, offset=off).astype(np.int64)
    off += 4 * n_sizes
    pairs = np.frombuffer(buf, dtype=_PAIR, count=int(sizes.sum()), offset=off)
    return CSR(
        indptr=torch.as_tensor(np.concatenate([[0], sizes.cumsum()])).to(torch.long),
        indices=torch.as_tensor(pairs["idx"].astype(np.int64)).to(torch.long),
        values=torch.as_tensor(pairs["val"].copy()).to(dtype),
        shape=(int(nrows), int(ncols)),
    ).to(device)


def write_bin_smat(mat: CSR, path: str) -> None:
    counts = mat.row_counts().cpu().numpy().astype(np.int32)
    pairs = np.empty(mat.nnz, dtype=_PAIR)
    pairs["idx"] = mat.indices.cpu().numpy().astype(np.int32)
    pairs["val"] = mat.values.detach().cpu().numpy().astype(np.float32)
    with open(path, "wb") as f:
        f.write(np.array([mat.nrows, mat.ncols], dtype=np.int32).tobytes())
        f.write(np.array([mat.nrows], dtype=np.int64).tobytes())
        f.write(counts.tobytes())
        f.write(pairs.tobytes())


# --------------------------------------------------------------------------- #
# binary vectors (write_vec_bin / read_vec_bin)
# --------------------------------------------------------------------------- #
def read_bin_vec(path: str, np_dtype=np.float32) -> np.ndarray:
    buf = np.fromfile(path, dtype=np.uint8)
    n = int(np.frombuffer(buf, dtype=np.int64, count=1, offset=0)[0])
    return np.frombuffer(buf, dtype=np_dtype, count=n, offset=8).copy()


def write_bin_vec(vec: np.ndarray, path: str) -> None:
    vec = np.asarray(vec, dtype=np.float32)
    with open(path, "wb") as f:
        f.write(np.array([vec.size], dtype=np.int64).tobytes())
        f.write(vec.tobytes())


# --------------------------------------------------------------------------- #
# misc
# --------------------------------------------------------------------------- #
def read_seen_labels(path: str) -> np.ndarray:
    with open(path) as f:
        return np.array([int(x) for x in f.read().split()], dtype=np.int64)


def write_seen_labels(labels, path: str) -> None:
    with open(path, "w") as f:
        for y in np.asarray(labels).tolist():
            f.write("%d\n" % y)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path
