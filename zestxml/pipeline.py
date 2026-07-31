"""The three stages of ZestXML, mirroring ``run.cpp``.

``xhtp_approx``     mine the sparsity pattern of W                 -> model_dir/*.bin
``xhtp_fine_tune``  shortlist + learn the bilinear classifier      -> model_dir/bilinear_clf.bin
``predict``         shortlist + score test points                  -> res_dir/score_mat.bin
"""

from __future__ import annotations

import os
import sys
import time
import numpy as np
import torch

from .csr import CSR
from .io import (
    ensure_dir,
    read_bin_smat,
    read_bin_vec,
    read_desc_file,
    read_seen_labels,
    read_text_smat,
    write_bin_smat,
    write_bin_vec,
    write_seen_labels,
)
from .model import (
    BilinearClassifier,
    BilinearPattern,
    direct_scores,
    get_shortlist,
    pair_targets,
)
from .params import DEFAULTS, INERT, Params
from .pattern import build_sparsity_pattern, union_pattern

SEP = os.sep


def log(msg: str = "") -> None:
    print(msg, flush=True)


class _Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *args):
        log("\nfinished in %.2f s" % (time.time() - self.t0))


def resolve_device(params: Params) -> torch.device:
    name = params.str("device")
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def _dtype(params: Params):
    return torch.float64 if params.bool("float64") else torch.float32


def _load(path: str, device, dtype) -> CSR:
    reader = read_bin_smat if path.endswith(".bin") else read_text_smat
    log("loading %s" % path)
    return reader(path, device=device, dtype=dtype)


def seen_labels_of(trn_X_Y: CSR) -> torch.Tensor:
    """Labels with at least one training point (``remove_test_labels``)."""
    freq = torch.zeros(trn_X_Y.ncols, dtype=torch.long, device=trn_X_Y.device)
    freq.index_add_(0, trn_X_Y.indices, torch.ones_like(trn_X_Y.indices))
    return (freq > 0).nonzero(as_tuple=True)[0]


def drop_unseen_labels(Y_Yf: CSR, seen: torch.Tensor) -> CSR:
    """Blank out the features of labels that have no training point."""
    mask = torch.ones(Y_Yf.nrows, dtype=torch.bool, device=Y_Yf.device)
    mask[seen] = False
    return Y_Yf.clear_rows(mask.nonzero(as_tuple=True)[0])


# --------------------------------------------------------------------------- #
# stage 1
# --------------------------------------------------------------------------- #
def run_xhtp_approx(params: Params) -> None:
    device, dtype = resolve_device(params), _dtype(params)
    model_dir = ensure_dir(params.path("model_dir"))

    trn_X_Xf = _load(params.path("trn_X_Xf"), device, dtype)
    Y_Yf = _load(params.path("Y_Yf"), device, dtype)
    trn_X_Y = _load(params.path("trn_X_Y"), device, dtype)
    Xf = read_desc_file(params.path("Xf"))
    Yf = read_desc_file(params.path("Yf"))

    log("bs_count         : %d" % params.int("bs_count"))
    log("bs_threshold     : %g" % params.float("bs_threshold"))
    log("bs_alpha         : %g" % params.float("bs_alpha"))
    log("bs_direct_wt     : %g" % params.float("bs_direct_wt"))

    with _Timer():
        Y_Yf = drop_unseen_labels(Y_Yf, seen_labels_of(trn_X_Y))
        Xf_Yf, Yf_Xf, direct = build_sparsity_pattern(
            trn_X_Xf,
            Y_Yf,
            trn_X_Y,
            Xf,
            Yf,
            bs_count=params.int("bs_count"),
            bs_alpha=params.float("bs_alpha"),
            bs_threshold=params.float("bs_threshold"),
            bs_direct_wt=params.float("bs_direct_wt"),
            max_elems=params.int("max_elems"),
            dense_elems=params.int("dense_elems"),
            log=log,
        )
        sparsity_pattern = union_pattern(Xf_Yf, Yf_Xf)

        log("[STAT] nnz of sparsity pattern mat : %d" % sparsity_pattern.nnz)
        log("[STAT] avg nnz of sparsity pattern mat per Xf : %.2f" % (sparsity_pattern.nnz / max(1, sparsity_pattern.nrows)))
        log("[STAT] avg nnz of sparsity pattern mat per Yf : %.2f" % (sparsity_pattern.nnz / max(1, sparsity_pattern.ncols)))
        log("[STAT] nnz of Xf_Yf mat : %d" % Xf_Yf.nnz)
        log("[STAT] nnz of Yf_Xf mat : %d" % Yf_Xf.nnz)

        write_bin_smat(Xf_Yf, model_dir + SEP + "Xf_Yf.bin")
        write_bin_smat(Yf_Xf, model_dir + SEP + "Yf_Xf.bin")
        write_bin_smat(sparsity_pattern, model_dir + SEP + "sparsity_pattern.bin")
        if direct is not None:
            write_bin_smat(direct, model_dir + SEP + "direct_Xf_Yf.bin")


# --------------------------------------------------------------------------- #
# stage 2
# --------------------------------------------------------------------------- #
def run_xhtp_fine_tune(params: Params) -> None:
    device, dtype = resolve_device(params), _dtype(params)
    model_dir = ensure_dir(params.path("model_dir"))

    trn_X_Xf = _load(params.path("trn_X_Xf"), device, dtype)
    Y_Yf = _load(params.path("Y_Yf"), device, dtype)
    trn_X_Y = _load(params.path("trn_X_Y"), device, dtype)

    log("loading sparsity pattern mat from model dir")
    sparsity_pattern = read_bin_smat(model_dir + SEP + "sparsity_pattern.bin", device, dtype)
    Xf_Yf = read_bin_smat(model_dir + SEP + "Xf_Yf.bin", device, dtype)
    Yf_Xf = read_bin_smat(model_dir + SEP + "Yf_Xf.bin", device, dtype)

    seen_file = model_dir + SEP + "seen_labels.txt"
    if os.path.exists(seen_file):
        log("using seen labels file : %s" % seen_file)
        seen = torch.as_tensor(read_seen_labels(seen_file)).to(device)
    else:
        log("generating seen labels from trn_X_Y")
        seen = seen_labels_of(trn_X_Y)
        write_seen_labels(seen.cpu().numpy(), seen_file)

    with _Timer():
        Y_Yf = drop_unseen_labels(Y_Yf, seen).unit_normalize_rows()
        trn_X_Xf = trn_X_Xf.unit_normalize_rows()

        log("\ngetting %d shortlist per point..." % params.int("shortyK"))
        shortlist = get_shortlist(
            trn_X_Xf,
            Y_Yf,
            sparsity_pattern,
            params.int("shortyK"),
            max_elems=params.int("max_elems"),
            dense_elems=params.int("dense_elems"),
            log=log,
        )
        write_bin_smat(shortlist, model_dir + SEP + "shortlist.bin")
        log("[STAT] nnz of shortlist    : %d" % shortlist.nnz)
        log("[STAT] recall of shortlist : %.2f%%" % shortlist.recall(trn_X_Y))

        log("\ntraining bilinear classifier")
        pattern = BilinearPattern.from_parts(Xf_Yf, Yf_Xf)
        clf = BilinearClassifier(
            pattern,
            kind=params.int("bilinear_classifier_kind"),
            cost=params.float("bilinear_classifier_cost"),
            pos_wt=params.float("bilinear_classifier_pos_wt"),
            normalize=params.bool("bilinear_normalize"),
            device=device,
            dtype=dtype,
        )
        log("classifier_cost : %.2f" % clf.cost)
        log("classifier_kind : %d" % clf.kind)
        log("[STAT] : nnz in assign_mat : %d" % shortlist.nnz)
        log("[STAT] : num parameters : %d" % (pattern.size + 1))

        targets = pair_targets(shortlist, trn_X_Y)
        clf.fit(
            trn_X_Xf,
            Y_Yf,
            shortlist,
            targets,
            epochs=params.int("bilinear_classifier_maxitr"),
            lr=params.float("lr"),
            batch_size=params.int("batch_size"),
            max_elems=params.int("max_elems"),
            seed=params.int("seed"),
            log=log,
        )
        clf.training_report(trn_X_Xf, Y_Yf, shortlist, targets, params.int("max_elems"), log=log)

        # same layout as the C++ model: the pattern weights followed by the bias
        weights = np.concatenate([clf.weights.cpu().numpy(), clf.bias.cpu().numpy()])
        write_bin_vec(weights, model_dir + SEP + "bilinear_clf.bin")


# --------------------------------------------------------------------------- #
# stage 3
# --------------------------------------------------------------------------- #
def run_predict(params: Params) -> None:
    device, dtype = resolve_device(params), _dtype(params)
    model_dir, res_dir = params.path("model_dir"), ensure_dir(params.path("res_dir"))

    tst_X_Xf = _load(params.path("tst_X_Xf"), device, dtype)
    Y_Yf = _load(params.path("Y_Yf"), device, dtype)
    tst_X_Y = _load(params.path("tst_X_Y"), device, dtype) if params.given("tst_X_Y") else None
    direct = read_bin_smat(model_dir + SEP + "direct_Xf_Yf.bin", device, dtype)

    if params.given("sparsity_pattern_file"):
        log("loading sparsity pattern mat from file : %s" % params.path("sparsity_pattern_file"))
        sparsity_pattern = _load(params.path("sparsity_pattern_file"), device, dtype)
        Xf_Yf, Yf_Xf = sparsity_pattern, CSR.empty((sparsity_pattern.ncols, sparsity_pattern.nrows), device, dtype)
    else:
        log("loading sparsity pattern mat from model dir")
        sparsity_pattern = read_bin_smat(model_dir + SEP + "sparsity_pattern.bin", device, dtype)
        Xf_Yf = read_bin_smat(model_dir + SEP + "Xf_Yf.bin", device, dtype)
        Yf_Xf = read_bin_smat(model_dir + SEP + "Yf_Xf.bin", device, dtype)

    log("loading model...")
    clf_vec = read_bin_vec(model_dir + SEP + "bilinear_clf.bin")

    with _Timer():
        tst_X_Xf = tst_X_Xf.unit_normalize_rows()
        Y_Yf = Y_Yf.unit_normalize_rows()

        log("\ngetting %d shortlist per point..." % params.int("shortyK"))
        shortlist = get_shortlist(
            tst_X_Xf,
            Y_Yf,
            sparsity_pattern,
            params.int("shortyK"),
            max_elems=params.int("max_elems"),
            dense_elems=params.int("dense_elems"),
            log=log,
        )
        log("[STAT] nnz of shortlist    : %d" % shortlist.nnz)
        if tst_X_Y is not None:
            log("[STAT] recall of shortlist : %.2f%%" % shortlist.recall(tst_X_Y))
        write_bin_smat(shortlist, res_dir + SEP + "shortlist.bin")

        pattern = BilinearPattern.from_parts(Xf_Yf, Yf_Xf)
        clf = BilinearClassifier(
            pattern,
            kind=params.int("bilinear_classifier_kind"),
            normalize=params.bool("bilinear_normalize"),
            device=device,
            dtype=dtype,
        )
        assert clf_vec.size == pattern.size + 1, (
            "model has %d weights but the pattern needs %d + 1" % (clf_vec.size, pattern.size)
        )
        clf.weights = torch.as_tensor(clf_vec[:-1]).to(device=device, dtype=dtype)
        clf.bias = torch.as_tensor(clf_vec[-1:]).to(device=device, dtype=dtype)

        max_elems = params.int("max_elems")
        bilinear = clf.score_matrix(tst_X_Xf, Y_Yf, shortlist, max_elems, transform=True)
        write_bin_smat(shortlist.with_values(bilinear), res_dir + SEP + "bilinear_score_mat.bin")

        knn = direct_scores(direct, tst_X_Xf, Y_Yf, shortlist, max_elems)
        write_bin_smat(shortlist.with_values(knn), res_dir + SEP + "knn_score_mat.bin")

        alpha = params.float("score_alpha")
        write_bin_smat(shortlist.with_values(alpha * bilinear + (1.0 - alpha) * knn), res_dir + SEP + "score_mat.bin")
        log("wrote %s" % (res_dir + SEP + "score_mat.bin"))


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    params = Params.parse(list(sys.argv[1:] if argv is None else argv))
    ensure_dir(params.path("res_dir"))
    params.dump(params.path("res_dir") + SEP + "params.txt")

    if params.int("num_thread") > 0:
        torch.set_num_threads(params.int("num_thread"))
    if params.bool("bilinear_add_bias"):
        raise NotImplementedError("bilinear_add_bias is not supported (unused by every run script)")
    for name, why in INERT.items():
        if name in params.values and params.values[name] != DEFAULTS.get(name):
            log("note: -%s has no effect -- %s" % (name, why))

    kind = params.str("type")
    device = resolve_device(params)
    log("running : %s on %s" % (kind, device))

    if kind in ("xhtp_approx", "train", "all"):
        run_xhtp_approx(params)
    if kind in ("xhtp_fine_tune", "train", "all"):
        run_xhtp_fine_tune(params)
    if kind in ("predict", "all"):
        run_predict(params)
    if kind not in ("xhtp_approx", "xhtp_fine_tune", "train", "predict", "all"):
        sys.stderr.write("unknown -type '%s'\n" % kind)
        return 1
    return 0
