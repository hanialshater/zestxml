"""Train the *lexical* model on candidates retrieved *semantically*.

    python benchmarks/rq_retrain.py GZXML-Datasets/GZ-Reuters-90 GZXML-Datasets/Rq-reu-0.10

An earlier experiment swapped a semantic shortlist into prediction and found that 12.5
extra points of unseen recall bought nothing. That test had a hole: it reused a model
trained on *lexically* retrieved negatives and then asked it to rank *semantically*
retrieved ones. The negative distribution changed between training and test, which is a
mismatch rather than a result.

This closes it. Both shortlists are regenerated -- training and test -- from the semantic
dataset, optionally unioned with the lexical ones, and the lexical model is trained from
scratch against them. The features never change; only the candidate set does, and it
changes consistently in both stages.

Prints four arms: the lexical control, semantic candidates, the union, and (for reference)
the semantic dataset scored on its own terms.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zestxml import ZestXML  # noqa: E402
from zestxml.csr import CSR  # noqa: E402
from zestxml.io import ensure_dir, read_bin_smat, read_text_smat, write_bin_smat  # noqa: E402

CONFIG = dict(bs_alpha=0.02, bs_direct_wt=0.8, bilinear_classifier_cost=5,
              bilinear_normalize=0, bilinear_classifier_maxitr=20, num_thread=0)


def union(a: CSR, b: CSR) -> CSR:
    """Candidate sets from two channels, merged per point.

    Done on the sparse keys rather than by densifying: on GZ-NPM a dense candidate matrix
    is 25127 x 3223 per operand, which is a few hundred megabytes for a result that has
    well under a million non-zeros.
    """
    assert a.shape == b.shape, f"{a.shape} != {b.shape}"
    ncols = a.shape[1]
    keys = torch.cat([a.row_ids() * ncols + a.indices, b.row_ids() * ncols + b.indices])
    keys = torch.unique(keys)
    rows, cols = keys // ncols, keys % ncols
    counts = torch.zeros(a.nrows, dtype=torch.long, device=rows.device)
    counts.index_add_(0, rows, torch.ones_like(rows))
    indptr = torch.cat([torch.zeros(1, dtype=torch.long, device=rows.device), counts.cumsum(0)])
    return CSR(indptr, cols.contiguous(), torch.ones(cols.numel()), a.shape)


def recalls(shortlist_path, truth, unseen_mask):
    sl = read_bin_smat(shortlist_path).to_dense() != 0
    t = truth.to_dense() > 0
    tu = t & unseen_mask[None, :]
    return (100.0 * (sl & t).sum().item() / t.sum().item(),
            100.0 * (sl & tu).sum().item() / tu.sum().item())


def main(lexical_dir, semantic_dir, out="Results", **cfg):
    config = {**CONFIG, **cfg}
    truth = read_text_smat(f"{lexical_dir}/tst_X_Y.txt")
    unseen = torch.zeros(truth.ncols, dtype=torch.bool)
    with open(f"{lexical_dir}/unseen_labels.txt") as f:
        for line in f:
            if line.strip():
                unseen[int(line.split()[0])] = True

    print("=== control: lexical features, lexical candidates")
    lex = ZestXML(lexical_dir, f"{out}/Rt-lex", **config)
    lex.fit(); lex.predict()
    rows = [("lexical control", lex.evaluate(verbose=False), f"{out}/Rt-lex")]

    print("\n=== semantic dataset, scored on its own terms (reference)")
    sem = ZestXML(semantic_dir, f"{out}/Rt-sem", **config)
    sem.fit(); sem.predict()
    rows.append(("semantic features", sem.evaluate(verbose=False), f"{out}/Rt-sem"))

    # the candidate sets the semantic model generated, for train and for test
    sem_trn, sem_tst = f"{out}/Rt-sem/model/shortlist.bin", f"{out}/Rt-sem/shortlist.bin"
    lex_trn, lex_tst = f"{out}/Rt-lex/model/shortlist.bin", f"{out}/Rt-lex/shortlist.bin"

    ensure_dir(f"{out}/Rt-union")
    uni_trn, uni_tst = f"{out}/Rt-union/trn_shortlist.bin", f"{out}/Rt-union/tst_shortlist.bin"
    write_bin_smat(union(read_bin_smat(sem_trn), read_bin_smat(lex_trn)), uni_trn)
    write_bin_smat(union(read_bin_smat(sem_tst), read_bin_smat(lex_tst)), uni_tst)

    for name, trn_sl, tst_sl, res in (
        ("lexical + sem candidates", sem_trn, sem_tst, f"{out}/Rt-semcand"),
        ("lexical + union candidates", uni_trn, uni_tst, f"{out}/Rt-unioncand"),
    ):
        print(f"\n=== {name} (features unchanged, candidates swapped in BOTH stages)")
        m = ZestXML(lexical_dir, res, **config,
                    trn_shortlist_file=trn_sl, shortlist_file=tst_sl)
        m.fit(); m.predict()
        rows.append((name, m.evaluate(verbose=False), res))

    print("\n" + "=" * 92)
    print("%-28s %7s %7s %9s %9s %9s %9s" % ("arm", "P@1", "PSP@5", "unseen", "seen", "recall", "uns-rec"))
    for name, mm, res in rows:
        r, ur = recalls(f"{res}/shortlist.bin", truth, unseen)
        print("%-28s %7.2f %7.2f %9.2f %9.2f %9.2f %9.2f" % (
            name, mm["all labels"]["P@1"], mm["all labels"]["PSP@5"],
            mm["unseen only"]["P@1"], mm["seen only"]["P@1"], r, ur))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("lexical_dir")
    ap.add_argument("semantic_dir")
    ap.add_argument("--out", default="Results")
    ap.add_argument("overrides", nargs="*", help="key=value run parameters, e.g. shortyK=100")
    a = ap.parse_args()
    main(a.lexical_dir, a.semantic_dir, a.out,
         **dict(o.split("=", 1) for o in a.overrides))
