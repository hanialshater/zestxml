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


def test_char_ngram_similarity_ranks_exact_match_first():
    from zestxml.embed import char_ngram_matrices, topk_cosine_sparse

    targets = ["housing", "housing starts", "warehousing", "unrelated"]
    left, right = char_ngram_matrices(["housing"], targets)
    sims = topk_cosine_sparse(left, right, topk=3, min_sim=0.1)
    best = sims.indices[sims.values.argmax()].item()
    assert targets[best] == "housing"
    assert abs(sims.values.max().item() - 1.0) < 1e-5


def test_direct_map_fuzzy_fills_gaps_without_touching_exact_matches():
    Xf = ["housing", "acquire", "unrelated"]
    Yf = ["1_housing", "1_housings", "__label__0__x"]

    exact = direct_map(Xf, Yf, 0.8)
    assert exact.nnz == 1  # only "1_housing" matches a point feature by name

    fuzzy = direct_map(Xf, Yf, 0.8, mode="charngram", topk=2, min_sim=0.3, log=lambda *a: None)
    pairs = dict(zip(zip(fuzzy.row_ids().tolist(), fuzzy.indices.tolist()), fuzzy.values.tolist()))
    assert pairs[(0, 0)] == pytest.approx(0.8)  # the exact link keeps full weight
    assert (0, 1) in pairs and pairs[(0, 1)] < 0.8  # "1_housings" reached "housing", discounted
    assert not any(col == 2 for _, col in pairs)  # per-label features are never fuzzy matched


def test_direct_map_fallback_only_adds_nothing_when_everything_matches():
    Xf = ["housing", "acquire"]
    Yf = ["1_housing"]
    fuzzy = direct_map(Xf, Yf, 0.8, mode="charngram", topk=2, min_sim=0.1, log=lambda *a: None)
    assert fuzzy.nnz == 1 and fuzzy.values.tolist() == [pytest.approx(0.8)]


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


# --------------------------------------------------------------------------- #
# the dataset API
# --------------------------------------------------------------------------- #
def test_build_dataset_end_to_end(tmp_path):
    """A dataset built from raw texts must load and train without any manual wiring."""
    from zestxml.dataset import build_dataset, select_unseen_labels
    from zestxml.io import read_text_smat, read_desc_file

    words = ["hiking", "boots", "camera", "lens", "coffee", "grinder", "laptop", "keyboard"]
    tags = ["hiking", "photography", "coffee", "computing"]
    g = torch.Generator().manual_seed(3)
    texts, labels = [], []
    for i in range(300):
        t = int(torch.randint(0, len(tags), (1,), generator=g))
        pair = words[2 * t : 2 * t + 2]
        texts.append(" ".join(pair * 3) + f" item{i % 17}")
        labels.append([tags[t]])

    out = str(tmp_path / "ds")
    stats = build_dataset(out, texts[:200], labels[:200], texts[200:], labels[200:],
                          label_names=tags, verbose=False)
    assert stats["labels"] == 4 and stats["train_points"] == 200

    # every matrix must agree with the raw-text row counts, the alignment bug class
    for mat, txt, n in (("trn_X_Xf", "trn_X", 200), ("tst_X_Xf", "tst_X", 100)):
        assert read_text_smat(f"{out}/{mat}.txt").nrows == n
        assert len(read_desc_file(f"{out}/{txt}.txt")) + 1 == n + 1

    # the label-feature convention: tokens are "1_<tok>" so the direct map can find them
    yf = read_desc_file(f"{out}/Yf.txt")
    assert "1_hiking" in yf and any(n.startswith("__label__") for n in yf)

    # and the whole pipeline runs on it, through the public API
    from zestxml import ZestXML
    model = ZestXML(out, f"{out}/res", shortyK=4, bs_count=5, bilinear_classifier_maxitr=3)
    scores = model.fit().predict()
    assert scores.shape == (100, 4) and scores.nnz > 0
    assert read_bin_smat(f"{out}/res/score_mat.bin").nnz == scores.nnz

    metrics = model.evaluate(verbose=False)
    assert 0.0 <= metrics["all labels"]["P@1"] <= 100.0
    assert metrics["all labels"]["points"] == 100


def test_label_expand_adds_features_without_disturbing_the_default(tmp_path):
    """The hook must be inert when unused, and land on exactly the labels it names."""
    import filecmp
    from zestxml.dataset import build_dataset
    from zestxml.io import read_desc_file, read_text_smat

    texts = ["hiking boots trail gear"] * 40 + ["camera lens tripod photo"] * 40
    labels = [["hiking"]] * 40 + [["photography"]] * 40
    args = dict(trn_texts=texts, trn_labels=labels, tst_texts=texts, tst_labels=labels,
                label_names=["hiking", "photography"], verbose=False)

    plain = str(tmp_path / "plain")
    build_dataset(plain, **args)

    # default None is byte-identical, so switching the feature on is opt-in in the
    # strongest sense: an existing pipeline cannot drift by upgrading
    again = str(tmp_path / "again")
    build_dataset(again, label_expand=None, **args)
    same, diff, err = filecmp.cmpfiles(plain, again, os.listdir(plain), shallow=False)
    assert not diff and not err, f"label_expand=None changed {diff + err}"

    # the hook sees the whole label set and the point vocabulary, and only widens the
    # labels it returns
    seen = {}

    def expander(names, name_tokens, vocab):
        seen["names"], seen["vocab"] = list(names), set(vocab)
        return {"hiking": ["trail"]}

    wide = str(tmp_path / "wide")
    stats = build_dataset(wide, label_expand=expander, **args)
    assert seen["names"] == ["hiking", "photography"] and "trail" in seen["vocab"]
    assert stats["expanded_features"] == 1 and stats["max_labels_per_expanded_feature"] == 1

    yf = read_desc_file(f"{wide}/Yf.txt")
    trail = yf.index("1_trail")
    rows = read_text_smat(f"{wide}/Y_Yf.txt")
    carries = [r for r in range(rows.nrows)
               if trail in rows.indices[rows.indptr[r]:rows.indptr[r + 1]].tolist()]
    assert carries == [0], "1_trail must land on hiking and on no other label"


def test_api_rejects_unknown_parameters():
    """A misspelled hyper-parameter must fail loudly, not be silently ignored."""
    from zestxml import ZestXML
    with pytest.raises(TypeError, match="shortyk"):
        ZestXML("nowhere", "nowhere/res", shortyk=5)


def test_select_unseen_labels_keeps_them_out_of_training_only():
    from zestxml.dataset import select_unseen_labels
    trn = [[0, 1], [0, 2], [1, 2]] * 20
    tst = [[0, 1, 2]] * 20
    unseen = select_unseen_labels(trn, tst, 3, stride=2, min_test=5)
    assert unseen and all(0 <= i < 3 for i in unseen)
