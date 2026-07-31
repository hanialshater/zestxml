"""Unit tests: every sparse primitive is checked against a dense reference."""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zestxml.csr import CSR, prod_dense_rows, ranges, spspmm  # noqa: E402
from zestxml.io import read_bin_smat, read_text_smat, write_bin_smat, write_text_smat  # noqa: E402
from zestxml.model import BilinearClassifier, BilinearPattern, get_shortlist, pair_scores, pair_targets  # noqa: E402
from zestxml.pattern import add_matrices, direct_map, remove_duplicates  # noqa: E402


def random_csr(rows, cols, density=0.1, seed=0, binary=False):
    g = torch.Generator().manual_seed(seed)
    dense = (torch.rand(rows, cols, generator=g) < density).float()
    if not binary:
        dense *= torch.rand(rows, cols, generator=g) + 0.1
    nz = dense.nonzero(as_tuple=False)
    return CSR.from_coo(nz[:, 0], nz[:, 1], dense[nz[:, 0], nz[:, 1]], (rows, cols)), dense


def test_ranges():
    starts = torch.tensor([5, 0, 2])
    counts = torch.tensor([2, 0, 3])
    assert ranges(starts, counts).tolist() == [5, 6, 2, 3, 4]


def test_csr_roundtrip_and_transpose():
    mat, dense = random_csr(17, 11, seed=1)
    assert torch.allclose(mat.to_dense(), dense)
    assert torch.allclose(mat.transpose().to_dense(), dense.t())
    assert torch.allclose(mat.transpose().transpose().to_dense(), dense)


def test_unit_normalize_rows():
    mat, dense = random_csr(9, 7, seed=2)
    norms = mat.unit_normalize_rows().to_dense().norm(dim=1)
    expected = torch.where(dense.norm(dim=1) > 0, torch.ones(9), torch.zeros(9))
    assert torch.allclose(norms, expected, atol=1e-6)


def test_products_match_dense():
    a, da = random_csr(13, 9, seed=3)
    b, db = random_csr(9, 15, seed=4)
    assert torch.allclose(spspmm(a, b).to_dense(), da @ db, atol=1e-5)
    rows = torch.arange(13)
    assert torch.allclose(prod_dense_rows(a, b, rows), da @ db, atol=1e-5)
    # chunking must not change the result
    assert torch.allclose(spspmm(a, b, max_elems=4).to_dense(), da @ db, atol=1e-5)


def test_add_matrices_keeps_cpp_entry_order():
    a = CSR.from_coo(torch.tensor([0, 0]), torch.tensor([1, 3]), torch.tensor([1.0, 2.0]), (2, 5))
    b = CSR.from_coo(torch.tensor([0, 0]), torch.tensor([0, 3]), torch.tensor([0.5, 0.5]), (2, 5))
    merged = add_matrices(a, b)
    # existing entries stay first and in place; new ones are appended after them
    assert merged.indices.tolist() == [1, 3, 0]
    assert torch.allclose(merged.values, torch.tensor([1.0, 2.5, 0.5]))
    assert torch.allclose(merged.to_dense(), a.to_dense() + b.to_dense())


def test_remove_duplicates():
    Xf_Yf = CSR.from_coo(torch.tensor([0, 2]), torch.tensor([1, 3]), torch.ones(2), (4, 5))
    Yf_Xf = CSR.from_coo(torch.tensor([1, 3]), torch.tensor([0, 1]), torch.ones(2), (5, 4))
    kept = remove_duplicates(Yf_Xf, Xf_Yf)  # (1, 0) transposes to (0, 1), already in Xf_Yf
    assert kept.nnz == 1
    assert kept.row_ids().tolist() == [3] and kept.indices.tolist() == [1]


def test_direct_map_prefix_rule():
    Xf = ["dog", "cat", "bird"]
    Yf = ["1_dog", "__label__7__x", "cat", "9_missing"]
    mat = direct_map(Xf, Yf, 0.8)
    pairs = sorted(zip(mat.row_ids().tolist(), mat.indices.tolist()))
    # "1_dog" -> "dog"; "cat" has no underscore so it matches whole; the rest do not match
    assert pairs == [(0, 0), (1, 2)]
    assert torch.allclose(mat.values, torch.full((2,), 0.8))


def test_binary_io_roundtrip(tmp_path):
    mat, _ = random_csr(23, 13, seed=5)
    path = str(tmp_path / "m.bin")
    write_bin_smat(mat, path)
    back = read_bin_smat(path)
    assert back.shape == mat.shape
    assert torch.allclose(back.to_dense(), mat.to_dense(), atol=1e-6)


def test_text_io_roundtrip(tmp_path):
    mat, _ = random_csr(19, 7, seed=6)
    path = str(tmp_path / "m.txt")
    write_text_smat(mat, path)
    back = read_text_smat(path)
    assert back.shape == mat.shape
    assert torch.allclose(back.to_dense(), mat.to_dense(), atol=1e-4)


# --------------------------------------------------------------------------- #
# the bilinear scorer
# --------------------------------------------------------------------------- #
def _reference_scores(X, Y, W_dense, pairs):
    rows, cols = pairs.row_ids(), pairs.indices
    return ((X.to_dense()[rows] @ W_dense) * Y.to_dense()[cols]).sum(1)


def test_pair_scores_match_dense_bilinear_form():
    X, _ = random_csr(20, 12, density=0.3, seed=7)
    Y, _ = random_csr(9, 8, density=0.4, seed=8)
    Xf_Yf, _ = random_csr(12, 8, density=0.2, seed=9)
    Yf_Xf, _ = random_csr(8, 12, density=0.2, seed=10)
    Yf_Xf = remove_duplicates(Yf_Xf, Xf_Yf)

    pattern = BilinearPattern.from_parts(Xf_Yf, Yf_Xf)
    weights = torch.randn(pattern.size, generator=torch.Generator().manual_seed(11))
    W_dense = pattern.as_csr(weights).to_dense()

    pairs, _ = random_csr(20, 9, density=0.5, seed=12)
    owner = pairs.row_ids()
    got = pair_scores(weights, pattern, X, Y, torch.arange(20), owner, pairs.indices)
    assert torch.allclose(got, _reference_scores(X, Y, W_dense, pairs), atol=1e-5)


def test_pair_scores_square_gives_linear_form_norms():
    X, _ = random_csr(15, 10, density=0.3, seed=13)
    Y, _ = random_csr(6, 7, density=0.4, seed=14)
    Xf_Yf, _ = random_csr(10, 7, density=0.3, seed=15)
    pattern = BilinearPattern.from_parts(Xf_Yf, CSR.empty((7, 10)))

    pairs, _ = random_csr(15, 6, density=0.5, seed=16)
    got = pair_scores(
        torch.ones(pattern.size), pattern, X, Y, torch.arange(15), pairs.row_ids(), pairs.indices, square=True
    )
    mask = pattern.as_csr(torch.ones(pattern.size)).to_dense()
    xd, yd = X.to_dense(), Y.to_dense()
    want = torch.stack(
        [((torch.outer(xd[i], yd[j]) * mask) ** 2).sum() for i, j in zip(pairs.row_ids(), pairs.indices)]
    )
    assert torch.allclose(got, want, atol=1e-5)


def test_pair_scores_batching_is_consistent():
    X, _ = random_csr(30, 12, density=0.3, seed=17)
    Y, _ = random_csr(11, 9, density=0.4, seed=18)
    Xf_Yf, _ = random_csr(12, 9, density=0.25, seed=19)
    pattern = BilinearPattern.from_parts(Xf_Yf, CSR.empty((9, 12)))
    weights = torch.randn(pattern.size, generator=torch.Generator().manual_seed(20))

    pairs, _ = random_csr(30, 11, density=0.4, seed=21)
    clf = BilinearClassifier(pattern)
    clf.weights = weights
    whole = clf.score_matrix(X, Y, pairs, max_elems=1 << 20)
    chunked = clf.score_matrix(X, Y, pairs, max_elems=8)
    assert torch.allclose(whole, chunked, atol=1e-6)


def test_shortlist_matches_bruteforce_topk():
    X, _ = random_csr(25, 10, density=0.4, seed=22)
    Y, _ = random_csr(13, 8, density=0.4, seed=23)
    SP, _ = random_csr(10, 8, density=0.3, seed=24)

    k = 5
    got = get_shortlist(X, Y, SP, k, log=lambda *a: None)
    want = X.to_dense() @ SP.to_dense() @ Y.to_dense().t()
    for i in range(X.nrows):
        lo, hi = got.indptr[i], got.indptr[i + 1]
        picked = got.indices[lo:hi]
        assert torch.allclose(got.values[lo:hi], want[i][picked], atol=1e-5)
        assert picked.tolist() == sorted(picked.tolist())
        cutoff = want[i][picked].min() if picked.numel() else float("inf")
        assert (want[i] > cutoff).sum() <= k


def test_pair_targets():
    pairs = CSR.from_coo(torch.tensor([0, 0, 1]), torch.tensor([1, 2, 0]), torch.ones(3), (2, 3))
    truth = CSR.from_coo(torch.tensor([0, 1]), torch.tensor([2, 0]), torch.ones(2), (2, 3))
    assert pair_targets(pairs, truth).tolist() == [-1.0, 1.0, 1.0]


def test_training_separates_positives():
    """A tiny fit must push true pairs above the decision boundary."""
    X, _ = random_csr(40, 12, density=0.4, seed=25)
    Y, _ = random_csr(8, 10, density=0.5, seed=26)
    Xf_Yf, _ = random_csr(12, 10, density=0.5, seed=27)
    pattern = BilinearPattern.from_parts(Xf_Yf, CSR.empty((10, 12)))

    pairs, _ = random_csr(40, 8, density=1.0, seed=28)
    truth, _ = random_csr(40, 8, density=0.15, seed=29)
    targets = pair_targets(pairs, truth)

    clf = BilinearClassifier(pattern, cost=10.0)
    clf.fit(X, Y, pairs, targets, epochs=30, lr=0.1, max_elems=1 << 20, log=lambda *a: None)
    margins = clf.score_matrix(X, Y, pairs, max_elems=1 << 20)
    assert margins[targets > 0].mean() > margins[targets < 0].mean()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
