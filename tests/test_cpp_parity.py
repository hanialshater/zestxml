"""End-to-end parity against the C++ reference.

Builds ``./run``, runs it on a small synthetic dataset, then checks that the PyTorch
implementation reproduces it:

1. the mined sparsity pattern (``Xf_Yf`` / ``Yf_Xf`` / ``sparsity_pattern``) entry for entry;
2. the bilinear, knn and blended scores, computed from the *C++* model and the *C++*
   shortlist -- which also pins down the layout of the learnt weight vector, since a
   misordered ``w`` would score garbage;
3. that the exact shortlist recalls at least as much as the C++ approximate one.

Skipped when a compiler or the reference binary is unavailable.

    python tests/test_cpp_parity.py     # run directly for a readable report
"""

import os
import shutil
import subprocess
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tests.make_synthetic import make  # noqa: E402
from zestxml.csr import CSR  # noqa: E402
from zestxml.io import read_bin_smat, read_bin_vec, read_desc_file, read_text_smat  # noqa: E402
from zestxml.model import BilinearClassifier, BilinearPattern, direct_scores, get_shortlist  # noqa: E402
from zestxml.params import Params  # noqa: E402
from zestxml.pattern import build_sparsity_pattern, union_pattern  # noqa: E402
from zestxml.pipeline import drop_unseen_labels, run_xhtp_approx, seen_labels_of  # noqa: E402

DATA = os.path.join(ROOT, "GZXML-Datasets", "Synth")
CPP_RES = os.path.join(ROOT, "Results", "SynthCpp")
CPP_MODEL = os.path.join(CPP_RES, "model")

ARGS = dict(
    bs_count="10",
    bs_alpha="0.02",
    bs_direct_wt="0.8",
    bs_threshold="0",
    shortyK="20",
    bilinear_classifier_cost="5",
    bilinear_normalize="0",
)
MAX_ELEMS = 1 << 22


def _cpp_available():
    return shutil.which(os.environ.get("CXX", "g++")) is not None


@pytest.fixture(scope="module")
def cpp_run():
    """Dataset + a completed C++ train/predict run."""
    if not _cpp_available():
        pytest.skip("no C++ compiler available")
    make(DATA)
    if not os.path.exists(os.path.join(ROOT, "run")):
        subprocess.run(["make"], cwd=ROOT, check=True, capture_output=True)

    os.makedirs(CPP_MODEL, exist_ok=True)
    cmd = [
        "./run",
        "-trn_X_Xf", f"{DATA}/trn_X_Xf.txt",
        "-tst_X_Xf", f"{DATA}/tst_X_Xf.txt",
        "-Y_Yf", f"{DATA}/Y_Yf.txt",
        "-trn_X_Y", f"{DATA}/trn_X_Y.txt",
        "-tst_X_Y", f"{DATA}/tst_X_Y.txt",
        "-Xf", f"{DATA}/Xf.txt",
        "-Yf", f"{DATA}/Yf.txt",
        "-res_dir", CPP_RES,
        "-model_dir", CPP_MODEL,
        "-num_thread", "1",
        "-type", "all",
    ]
    for k, v in ARGS.items():
        cmd += ["-" + k, v]
    subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True)
    return CPP_RES


def _params(**over):
    values = dict(ARGS)
    values.update(
        trn_X_Xf=f"{DATA}/trn_X_Xf.txt",
        tst_X_Xf=f"{DATA}/tst_X_Xf.txt",
        Y_Yf=f"{DATA}/Y_Yf.txt",
        trn_X_Y=f"{DATA}/trn_X_Y.txt",
        tst_X_Y=f"{DATA}/tst_X_Y.txt",
        Xf=f"{DATA}/Xf.txt",
        Yf=f"{DATA}/Yf.txt",
    )
    values.update(over)
    argv = []
    for k, v in values.items():
        argv += ["-" + k, str(v)]
    return Params.parse(argv)


def assert_matches_up_to_ties(got: CSR, want: CSR, name, max_frac=0.02):
    """Compare mined patterns, allowing the top-k cut to fall differently among ties.

    ``prod_for_jaccard`` truncates with ``std::sort``, which is unstable, so when several
    feature pairs share the k-th best score either implementation may keep either one.
    Shared entries still have to agree numerically.
    """
    assert got.shape == want.shape, f"{name}: shape {got.shape} != {want.shape}"
    g_keys, g_order = torch.sort(got.keys())
    w_keys, w_order = torch.sort(want.keys())
    shared = torch.isin(g_keys, w_keys)
    gv = got.values[g_order][shared]
    wv = want.values[w_order][torch.isin(w_keys, g_keys)]
    assert torch.allclose(gv, wv, atol=1e-4, rtol=1e-3), (
        f"{name}: shared values differ by up to {(gv - wv).abs().max():.3g}"
    )
    differing = int((~shared).sum()) + int((~torch.isin(w_keys, g_keys)).sum())
    frac = differing / max(1, got.nnz + want.nnz)
    assert frac <= max_frac, f"{name}: {frac:.1%} of entries differ (nnz {got.nnz} vs {want.nnz})"
    print(f"\n{name}: nnz {got.nnz} vs {want.nnz}, {differing} entries differ at top-k ties")


def assert_same_matrix(got: CSR, want: CSR, name, atol=1e-4, rtol=1e-3):
    assert got.shape == want.shape, f"{name}: shape {got.shape} != {want.shape}"
    assert got.nnz == want.nnz, f"{name}: nnz {got.nnz} != {want.nnz}"
    g_keys, g_order = torch.sort(got.keys())
    w_keys, w_order = torch.sort(want.keys())
    assert torch.equal(g_keys, w_keys), f"{name}: different support"
    assert torch.allclose(got.values[g_order], want.values[w_order], atol=atol, rtol=rtol), (
        f"{name}: values differ by up to "
        f"{(got.values[g_order] - want.values[w_order]).abs().max():.3g}"
    )


# --------------------------------------------------------------------------- #
def test_sparsity_pattern_matches_cpp(cpp_run):
    trn_X_Xf = read_text_smat(f"{DATA}/trn_X_Xf.txt")
    Y_Yf = read_text_smat(f"{DATA}/Y_Yf.txt")
    trn_X_Y = read_text_smat(f"{DATA}/trn_X_Y.txt")
    Y_Yf = drop_unseen_labels(Y_Yf, seen_labels_of(trn_X_Y))

    Xf_Yf, Yf_Xf, direct = build_sparsity_pattern(
        trn_X_Xf,
        Y_Yf,
        trn_X_Y,
        read_desc_file(f"{DATA}/Xf.txt"),
        read_desc_file(f"{DATA}/Yf.txt"),
        bs_count=int(ARGS["bs_count"]),
        bs_alpha=float(ARGS["bs_alpha"]),
        bs_threshold=float(ARGS["bs_threshold"]),
        bs_direct_wt=float(ARGS["bs_direct_wt"]),
        max_elems=MAX_ELEMS,
        dense_elems=MAX_ELEMS,
        log=lambda *a: None,
    )

    # the direct map has no ties to break: it must match exactly
    assert_same_matrix(direct, read_bin_smat(f"{CPP_MODEL}/direct_Xf_Yf.bin"), "direct_Xf_Yf")
    assert_matches_up_to_ties(Xf_Yf, read_bin_smat(f"{CPP_MODEL}/Xf_Yf.bin"), "Xf_Yf")
    assert_matches_up_to_ties(Yf_Xf, read_bin_smat(f"{CPP_MODEL}/Yf_Xf.bin"), "Yf_Xf")
    assert_matches_up_to_ties(
        union_pattern(Xf_Yf, Yf_Xf), read_bin_smat(f"{CPP_MODEL}/sparsity_pattern.bin"), "sparsity_pattern"
    )


def cpp_dropped_pairs(X: CSR, Y: CSR, W_dense, pairs: CSR):
    """Pairs the C++ scores wrong because of the zero sentinel in ``prod_helper``.

    ``prod_helper`` (``Source/mat.h``) records an accumulator slot with
    ``if (sum[id] == 0) indices.push_back(id)``, so a label feature whose running sum is
    still exactly zero when a second point feature touches it gets pushed twice. The
    write loop in ``prod`` then stores the real value for the first copy and zeroes
    ``sum[id]``, leaving the duplicate holding 0 -- and ``sparse_prod`` assigns
    ``mask[yf]`` entry by entry, so the trailing zero wins and the whole contribution of
    that label feature is lost. Weights left at exactly 0 by dual coordinate descent make
    this reachable on real models, so this port does not reproduce it.
    """
    Xd, Yd = X.to_dense(), Y.to_dense()
    bad_cells = torch.zeros(X.nrows, Y.ncols, dtype=torch.bool)
    for point in range(X.nrows):
        feats = X.indices[X.indptr[point] : X.indptr[point + 1]]
        if feats.numel() == 0:
            continue
        feats, _ = torch.sort(feats)  # the C++ scans a point's features in index order
        contrib = W_dense[feats] * Xd[point, feats][:, None]
        touched = contrib != 0
        prefix = torch.cat([torch.zeros(1, contrib.shape[1]), contrib.cumsum(0)[:-1]])
        first = torch.where(touched.any(0), touched.float().argmax(0), torch.zeros(1, dtype=torch.long))
        pos = torch.arange(feats.numel())[:, None]
        bad_cells[point] = ((prefix == 0) & touched & (pos > first[None, :])).any(0)

    rows, cols = pairs.row_ids(), pairs.indices
    return ((Yd[cols] != 0) & bad_cells[rows]).any(1)


def test_weight_vector_layout_matches_cpp(cpp_run):
    """The C++ ``w``, laid out over our pattern, must reproduce the C++ scores."""
    Xf_Yf = read_bin_smat(f"{CPP_MODEL}/Xf_Yf.bin")
    Yf_Xf = read_bin_smat(f"{CPP_MODEL}/Yf_Xf.bin")
    pattern = BilinearPattern.from_parts(Xf_Yf, Yf_Xf)
    clf_vec = read_bin_vec(f"{CPP_MODEL}/bilinear_clf.bin")
    assert clf_vec.size == pattern.size + 1

    tst_X_Xf = read_text_smat(f"{DATA}/tst_X_Xf.txt").unit_normalize_rows()
    Y_Yf = read_text_smat(f"{DATA}/Y_Yf.txt").unit_normalize_rows()
    shortlist = read_bin_smat(f"{CPP_RES}/shortlist.bin")  # score the C++ shortlist itself

    clf = BilinearClassifier(pattern, kind=0, normalize=False)
    clf.weights = torch.as_tensor(clf_vec[:-1])
    clf.bias = torch.as_tensor(clf_vec[-1:])

    bilinear = clf.score_matrix(tst_X_Xf, Y_Yf, shortlist, MAX_ELEMS, transform=True)
    want = read_bin_smat(f"{CPP_RES}/bilinear_score_mat.bin")
    dropped = cpp_dropped_pairs(tst_X_Xf, Y_Yf, pattern.as_csr(clf.weights).to_dense(), shortlist)
    assert dropped.float().mean() < 0.01, "the reference dropped an implausible number of pairs"
    print(f"\n{int(dropped.sum())}/{shortlist.nnz} pairs hit the C++ prod_helper zero sentinel")

    order = torch.argsort(shortlist.keys())  # both matrices share the shortlist's support
    assert torch.equal(torch.sort(want.keys())[0], shortlist.keys()[order])
    ours, theirs = bilinear[order], want.values[torch.argsort(want.keys())]
    keep = ~dropped[order]
    assert torch.allclose(ours[keep], theirs[keep], atol=1e-5, rtol=1e-3), (
        f"max diff {(ours[keep] - theirs[keep]).abs().max():.3g}"
    )

    direct = read_bin_smat(f"{CPP_MODEL}/direct_Xf_Yf.bin")
    knn = direct_scores(direct, tst_X_Xf, Y_Yf, shortlist, MAX_ELEMS)
    assert_same_matrix(shortlist.with_values(knn), read_bin_smat(f"{CPP_RES}/knn_score_mat.bin"), "knn_score_mat")

    alpha = 0.9
    blended = read_bin_smat(f"{CPP_RES}/score_mat.bin").values[torch.argsort(read_bin_smat(f"{CPP_RES}/score_mat.bin").keys())]
    mine = (alpha * bilinear + (1 - alpha) * knn)[order]
    assert torch.allclose(mine[keep], blended[keep], atol=1e-5, rtol=1e-3)


def test_exact_shortlist_recalls_at_least_as_much(cpp_run):
    tst_X_Xf = read_text_smat(f"{DATA}/tst_X_Xf.txt").unit_normalize_rows()
    Y_Yf = read_text_smat(f"{DATA}/Y_Yf.txt").unit_normalize_rows()
    tst_X_Y = read_text_smat(f"{DATA}/tst_X_Y.txt")
    sparsity_pattern = read_bin_smat(f"{CPP_MODEL}/sparsity_pattern.bin")

    ours = get_shortlist(tst_X_Xf, Y_Yf, sparsity_pattern, int(ARGS["shortyK"]), log=lambda *a: None)
    theirs = read_bin_smat(f"{CPP_RES}/shortlist.bin")
    assert ours.recall(tst_X_Y) >= theirs.recall(tst_X_Y) - 1e-6
    overlap = ours.emultiply_nnz(theirs) / max(1, theirs.nnz)
    assert overlap > 0.9, f"only {overlap:.1%} of the C++ shortlist was reproduced"


def test_stage1_writes_cpp_readable_model(cpp_run, tmp_path):
    """Our stage 1 output must load in the C++ binary and drive its predict."""
    model_dir = str(tmp_path / "model")
    res_dir = str(tmp_path / "res")
    os.makedirs(model_dir)
    os.makedirs(res_dir)
    run_xhtp_approx(_params(model_dir=model_dir, res_dir=res_dir, type="xhtp_approx"))

    cmd = ["./run"]
    for k, v in {
        "trn_X_Xf": f"{DATA}/trn_X_Xf.txt",
        "tst_X_Xf": f"{DATA}/tst_X_Xf.txt",
        "Y_Yf": f"{DATA}/Y_Yf.txt",
        "trn_X_Y": f"{DATA}/trn_X_Y.txt",
        "tst_X_Y": f"{DATA}/tst_X_Y.txt",
        "Xf": f"{DATA}/Xf.txt",
        "Yf": f"{DATA}/Yf.txt",
        "res_dir": res_dir,
        "model_dir": model_dir,
        "num_thread": "1",
        "type": "xhtp_fine_tune",
        **ARGS,
    }.items():
        cmd += ["-" + k, v]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert os.path.exists(os.path.join(model_dir, "bilinear_clf.bin"))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
