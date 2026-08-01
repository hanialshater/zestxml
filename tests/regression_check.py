"""Re-run a recorded run and diff it against its stored artifacts, bit for bit.

    python tests/regression_check.py Results/Npm2exact Results/Regress-npm

Reads ``<reference>/params.txt``, re-runs those exact parameters with the current code
into a fresh directory, and compares the score matrices -- the values, not only the
metrics, since two runs can agree on P@1 while disagreeing on most of the matrix. Exits
non-zero on any mismatch.

This is the check that a refactor did not change behaviour. It is deliberately not part of
the pytest suite: it needs a real dataset and a recorded run, neither of which ships.

**Training is only bit-reproducible at ``-num_thread 1``.** Above that, torch's CPU
reductions accumulate in nondeterministic order, the difference compounds across Adam
steps, and two runs of identical code on identical data land ~2e-2 apart in the *bilinear*
scores (the mined pattern, the shortlist and the untrained knn term stay bit-identical, so
a difference in either of those is a real regression). Pass/fail therefore rests on the
support being identical, the untrained stages being bit-identical, and the metrics agreeing
to ``METRIC_TOLERANCE``, which sits above the measured noise floor. The trained-score
deltas are printed for inspection but cannot be a criterion.
"""

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from zestxml.eval import COLUMNS, report  # noqa: E402
from zestxml.io import read_bin_smat  # noqa: E402

# parameters that name a location rather than describe the computation
LOCATION = {"res_dir", "model_dir"}

# Stages that do not depend on the trained weights. These are deterministic whatever the
# thread count, so any difference in them is a real regression. The trained scores are
# reported but cannot be a pass/fail criterion: the recorded reference was produced on
# some thread count we no longer control, and the noise floor is not a fixed number.
EXACT = {"shortlist.bin", "knn_score_mat.bin"}

# Metrics derive from the trained scores and inherit their nondeterminism. Measured floor
# on GZ-NPM: two runs of identical code differ by up to 0.011, and a run differs from a
# reference recorded at another thread count by up to 0.022. Anything under this is noise;
# a real regression moves P@1 by far more.
METRIC_TOLERANCE = 0.05


def load_params(path):
    out = {}
    with open(path) as f:
        for line in f:
            parts = line.split(None, 1)
            if parts:
                out[parts[0]] = parts[1].strip() if len(parts) > 1 else ""
    return out


def rerun(params, res_dir):
    argv = [sys.executable, "run_torch.py"]
    for key, value in params.items():
        if key not in LOCATION:
            argv += ["-" + key, value]
    argv += ["-res_dir", res_dir, "-model_dir", res_dir + "/model", "-type", "all"]
    subprocess.run(argv, check=True, stdout=subprocess.DEVNULL)


def compare(ref_dir, new_dir, data_dir):
    ok = True
    for name in ("shortlist.bin", "bilinear_score_mat.bin", "knn_score_mat.bin", "score_mat.bin"):
        ref_path, new_path = f"{ref_dir}/{name}", f"{new_dir}/{name}"
        if not os.path.exists(ref_path):
            print(f"  {name:24s} SKIP (no reference)")
            continue
        ref, new = read_bin_smat(ref_path), read_bin_smat(new_path)
        if ref.shape != new.shape or ref.nnz != new.nnz:
            print(f"  {name:24s} FAIL shape/nnz {ref.shape}/{ref.nnz} vs {new.shape}/{new.nnz}")
            ok = False
            continue
        same_support = torch.equal(ref.indices, new.indices) and torch.equal(ref.indptr, new.indptr)
        delta = (ref.values - new.values).abs().max().item() if same_support else float("nan")
        if name in EXACT:
            verdict = "OK  " if same_support and delta < 1e-5 else "FAIL"
            ok = ok and verdict == "OK  "
        else:
            # informational: only bit-reproducible at -num_thread 1
            verdict = "INFO" if same_support else "FAIL"
            ok = ok and same_support
        print(f"  {name:24s} {verdict} support {'identical' if same_support else 'DIFFERS'}"
              f", max |delta| {delta:.3e}")

    ref_m, new_m = report(f"{ref_dir}/score_mat.bin", data_dir, verbose=False), \
        report(f"{new_dir}/score_mat.bin", data_dir, verbose=False)
    for split in ref_m:
        worst = max(abs(ref_m[split][c] - new_m[split][c]) for c in COLUMNS if c in ref_m[split])
        verdict = "OK  " if worst < METRIC_TOLERANCE else "FAIL"
        ok = ok and verdict == "OK  "
        print("  %-24s %s worst %+.4f  %s" % (f"metrics [{split}]", verdict, worst,
                                              " ".join("%s %.2f" % (c, new_m[split][c])
                                                       for c in COLUMNS if c in new_m[split])))
    return ok


if __name__ == "__main__":
    reference, target = sys.argv[1], sys.argv[2]
    params = load_params(f"{reference}/params.txt")
    data_dir = os.path.dirname(params["tst_X_Y"])
    print(f"{reference} -> {target}  (data {data_dir})")
    rerun(params, target)
    sys.exit(0 if compare(reference, target, data_dir) else 1)
