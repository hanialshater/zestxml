"""XC metrics: P@k, nDCG@k and propensity-scored PSP@k, split by seen / unseen labels.

    from zestxml.eval import report
    report("Results/MyData/score_mat.bin", "GZXML-Datasets/MyData")

The unseen split is the point of the exercise: a label with no training positive can only
be predicted through the text of its name, so its accuracy is what separates a zero-shot
method from a per-label classifier. Unseen labels come from ``unseen_labels.txt`` when the
dataset ships one and are derived from ``trn_X_Y`` otherwise. A ``pos_trn_tst.txt`` filter
matrix, if present, is applied before scoring.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .csr import CSR
from .io import read_bin_smat, read_text_smat

KS = (1, 3, 5)
COLUMNS = ["P@1", "P@3", "P@5", "nDCG@5", "PSP@1", "PSP@3", "PSP@5"]


def inv_propensity(trn_X_Y: CSR, A: float = 0.55, B: float = 1.5) -> torch.Tensor:
    freq = torch.zeros(trn_X_Y.ncols).index_add_(0, trn_X_Y.indices, torch.ones(trn_X_Y.nnz))
    C = (np.log(max(trn_X_Y.nrows, 2)) - 1) * (B + 1) ** A
    return 1.0 + C * torch.exp(-A * torch.log(freq + B))


def evaluate(
    scores: torch.Tensor,
    truth: torch.Tensor,
    inv_prop: torch.Tensor,
    ks: Sequence[int] = KS,
    label_mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """``scores`` and ``truth`` dense (num_points x num_labels)."""
    if label_mask is not None:
        scores = scores.clone()
        scores[:, ~label_mask] = -np.inf
        truth = truth * label_mask[None, :]

    ks = [k for k in ks if k <= scores.shape[1]]  # P@5 is undefined with 4 labels
    assert ks, "no cutoff fits: the dataset has fewer labels than the smallest k"

    keep = truth.sum(1) > 0  # points with nothing to retrieve carry no information
    scores, truth = scores[keep], truth[keep]
    out: Dict[str, float] = {}
    top = torch.topk(scores, max(ks), dim=1).indices
    hits = torch.gather(truth, 1, top)
    gain = torch.gather(inv_prop[None, :].expand_as(truth), 1, top) * hits

    for k in ks:
        out[f"P@{k}"] = 100.0 * hits[:, :k].sum(1).div(k).mean().item()

        discount = 1.0 / torch.log2(torch.arange(k, dtype=torch.float) + 2)
        dcg = (hits[:, :k] * discount[None, :]).sum(1)
        n_true = truth.sum(1).clamp(max=k).long()
        ideal = torch.cat([torch.zeros(1), discount.cumsum(0)])[n_true]
        out[f"nDCG@{k}"] = 100.0 * (dcg / ideal).mean().item()

        # PSP@k: achieved propensity weighted gain over the best achievable one
        best = torch.sort(truth * inv_prop[None, :], dim=1, descending=True).values[:, :k]
        out[f"PSP@{k}"] = 100.0 * gain[:, :k].sum().item() / max(best.sum().item(), 1e-9)
    out["points"] = int(keep.sum())
    return out


def unseen_labels(data_dir: str, trn: CSR, n_labels: int) -> torch.Tensor:
    """Labels with no training point, from the dataset's own file or derived."""
    unseen = torch.zeros(n_labels, dtype=torch.bool)
    path = f"{data_dir}/unseen_labels.txt"
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                if line.strip():
                    unseen[int(line.split()[0])] = True
        return unseen
    freq = torch.zeros(n_labels).index_add_(0, trn.indices, torch.ones(trn.nnz))
    return freq == 0


# Names the GZXML datasets have used for the pairs to zero before scoring. The reference
# metrics.py picks pos_trn_val.txt or pos_trn_tst.txt depending on which split it scores,
# so which one applies depends on `split`.
FILTER_NAMES = {"tst": ["pos_trn_tst.txt", "filter_labels_test.txt", "tst_filter_labels.txt"],
                "val": ["pos_trn_val.txt", "filter_labels_val.txt", "val_filter_labels.txt"]}


def report(
    scores: Union[str, CSR, torch.Tensor],
    data_dir: str,
    verbose: bool = True,
    split: str = "tst",
) -> Dict[str, Dict[str, float]]:
    """Evaluate a score matrix against a dataset directory.

    ``scores`` is a path to a ``.bin`` score matrix, a :class:`CSR`, or a dense tensor.
    Returns ``{"all labels": {...}, "unseen only": {...}, "seen only": {...}}``; the last
    two are present only when the dataset has both kinds of label.
    """
    truth = read_text_smat(f"{data_dir}/{split}_X_Y.txt")
    trn = read_text_smat(f"{data_dir}/trn_X_Y.txt")
    if isinstance(scores, str):
        # Two splits of the same dataset often have the same number of points, so a shape
        # check cannot tell you that a score matrix belongs to a different one -- scoring
        # test predictions against validation labels passes silently and reads as a
        # catastrophic result. The run's params.txt records which split it predicted.
        params = os.path.join(os.path.dirname(scores), "params.txt")
        if os.path.exists(params):
            for line in open(params):
                if line.startswith("tst_X_Y "):
                    got = os.path.basename(line.split(None, 1)[1].strip())
                    if got not in ("-", f"{split}_X_Y.txt") and verbose:
                        print(f"WARNING: {scores} was predicted against {got}, but you are "
                              f"scoring it against {split}_X_Y.txt. Re-run predict with "
                              f"-tst_X_Xf {data_dir}/{split}_X_Xf.txt to score this split.")
        scores = read_bin_smat(scores)
    if isinstance(scores, CSR):
        assert scores.shape == truth.shape, f"{scores.shape} != {truth.shape}"
        scores = scores.to_dense()
    dense_scores, dense_truth = scores.cpu().float(), (truth.to_dense() > 0).float()

    # A filter matrix removes (point, label) pairs before scoring -- typically train/test
    # overlap. Not applying one inflates every metric, so say which was used, and say so
    # just as loudly when none was found: a silent absence is how a number gets compared
    # against a published one that was filtered.
    # build_dataset writes empty *_filter_labels.txt placeholders, so existence is not
    # enough -- an empty file is "no filter", not a filter with no pairs.
    found = [n for n in FILTER_NAMES.get(split, [])
             if os.path.exists(f"{data_dir}/{n}") and os.path.getsize(f"{data_dir}/{n}") > 0]
    if found:
        drop = read_text_smat(f"{data_dir}/{found[0]}")
        dense_scores = dense_scores.clone()
        dense_scores[drop.row_ids(), drop.indices] = 0.0
        if verbose:
            print(f"applied filter matrix {found[0]} ({drop.nnz} pairs)"
                  + (f"; also present but unused: {found[1:]}" if len(found) > 1 else ""))
    elif verbose:
        print(f"no filter matrix found for split '{split}' (looked for "
              f"{', '.join(FILTER_NAMES.get(split, []))}) -- metrics are unfiltered")

    inv_prop = inv_propensity(trn)
    unseen = unseen_labels(data_dir, trn, truth.ncols)

    out = {"all labels": evaluate(dense_scores, dense_truth, inv_prop)}
    if 0 < int(unseen.sum()) < truth.ncols:
        out["unseen only"] = evaluate(dense_scores, dense_truth, inv_prop, label_mask=unseen)
        out["seen only"] = evaluate(dense_scores, dense_truth, inv_prop, label_mask=~unseen)
    if verbose:
        print(f"{int(unseen.sum())}/{truth.ncols} labels have no training point")
        print(format_report(out))
    return out


def format_report(rows: Dict[str, Dict[str, float]]) -> str:
    first = next(iter(rows.values()), {})
    cols = [c for c in COLUMNS if c in first]  # small datasets have no P@5
    lines: List[str] = ["%-12s %7s " % ("", "points") + " ".join("%7s" % c for c in cols)]
    for name, m in rows.items():
        lines.append("%-12s %7d " % (name, m["points"]) + " ".join("%7.2f" % m[c] for c in cols))
    return "\n".join(lines)
