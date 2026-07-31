"""Standalone XC metrics, so results can be checked without pyxclib.

    python tools/eval_xc.py Results/<dataset>/score_mat.bin GZXML-Datasets/<dataset>

Reports P@k and nDCG@k, propensity scored PSP@k (same propensity model as
``ips_weight`` in ``Source/helper.cpp``), and -- when the dataset carries an
``unseen_labels.txt`` -- the same numbers restricted to labels that had no training
point, which is the number the generalized zero-shot setting is actually about.
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zestxml.io import read_bin_smat, read_text_smat  # noqa: E402


def inv_propensity(trn_X_Y, A=0.55, B=1.5):
    freq = torch.zeros(trn_X_Y.ncols).index_add_(0, trn_X_Y.indices, torch.ones(trn_X_Y.nnz))
    C = (np.log(max(trn_X_Y.nrows, 2)) - 1) * (B + 1) ** A
    return 1.0 + C * torch.exp(-A * torch.log(freq + B))


def evaluate(scores, truth, inv_prop, ks=(1, 3, 5), label_mask=None):
    """``scores`` and ``truth`` dense (num_points x num_labels)."""
    if label_mask is not None:
        scores = scores.clone()
        scores[:, ~label_mask] = -np.inf
        truth = truth * label_mask[None, :]

    keep = truth.sum(1) > 0  # points with nothing to retrieve carry no information
    scores, truth = scores[keep], truth[keep]
    out = {}
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


def main(score_path, data_dir):
    truth = read_text_smat(f"{data_dir}/tst_X_Y.txt")
    trn = read_text_smat(f"{data_dir}/trn_X_Y.txt")
    scores = read_bin_smat(score_path)
    assert scores.shape == truth.shape, f"{scores.shape} != {truth.shape}"

    dense_scores, dense_truth = scores.to_dense(), (truth.to_dense() > 0).float()
    inv_prop = inv_propensity(trn)

    rows = [("all labels", evaluate(dense_scores, dense_truth, inv_prop))]
    unseen_file = f"{data_dir}/unseen_labels.txt"
    if os.path.exists(unseen_file):
        unseen = torch.zeros(truth.ncols, dtype=torch.bool)
        with open(unseen_file) as f:
            for line in f:
                if line.strip():
                    unseen[int(line.split()[0])] = True
        rows.append(("unseen only", evaluate(dense_scores, dense_truth, inv_prop, label_mask=unseen)))
        rows.append(("seen only", evaluate(dense_scores, dense_truth, inv_prop, label_mask=~unseen)))

    cols = ["P@1", "P@3", "P@5", "nDCG@5", "PSP@1", "PSP@3", "PSP@5"]
    print(f"{score_path}")
    print("%-12s %7s " % ("", "points") + " ".join("%7s" % c for c in cols))
    for name, m in rows:
        print("%-12s %7d " % (name, m["points"]) + " ".join("%7.2f" % m[c] for c in cols))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
