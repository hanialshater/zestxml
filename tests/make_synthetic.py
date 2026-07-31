"""Generate a small dataset in the GZXML text format, for tests and smoke runs.

    python tests/make_synthetic.py GZXML-Datasets/Synth

The C++ reference divides by ``dim / 1000`` when building its progress bar, so it
crashes on matrices with fewer than 1000 rows; the defaults here stay above that so the
same data can be fed to both implementations.
"""

import os
import sys

import numpy as np

NUM_TRN, NUM_TST = 2000, 1200
NUM_Y = 200
NUM_XF, NUM_YF = 1000, 400


def _write(path, rows, nrows, ncols):
    with open(path, "w") as f:
        f.write("%d %d\n" % (nrows, ncols))
        for row in rows:
            f.write(" ".join("%d:%.5f" % (i, v) for i, v in sorted(row)) + "\n")


def _rand_rows(rng, n, dim, lo, hi, binary=False):
    rows = []
    for _ in range(n):
        k = rng.randint(lo, hi + 1)
        idx = rng.choice(dim, size=k, replace=False)
        val = np.ones(k) if binary else rng.rand(k).astype(np.float32) + 0.1
        rows.append(list(zip(idx.tolist(), val.tolist())))
    return rows


def make(out_dir, seed=7):
    rng = np.random.RandomState(seed)
    os.makedirs(out_dir, exist_ok=True)

    _write(f"{out_dir}/trn_X_Xf.txt", _rand_rows(rng, NUM_TRN, NUM_XF, 5, 20), NUM_TRN, NUM_XF)
    _write(f"{out_dir}/tst_X_Xf.txt", _rand_rows(rng, NUM_TST, NUM_XF, 5, 20), NUM_TST, NUM_XF)
    _write(f"{out_dir}/Y_Yf.txt", _rand_rows(rng, NUM_Y, NUM_YF, 3, 10), NUM_Y, NUM_YF)
    _write(f"{out_dir}/trn_X_Y.txt", _rand_rows(rng, NUM_TRN, NUM_Y, 1, 4, True), NUM_TRN, NUM_Y)
    _write(f"{out_dir}/tst_X_Y.txt", _rand_rows(rng, NUM_TST, NUM_Y, 1, 4, True), NUM_TST, NUM_Y)

    with open(f"{out_dir}/Xf.txt", "w") as f:
        for i in range(NUM_XF):
            f.write("tok%d\n" % i)
    # every third label feature is named after a point feature, exercising the direct map
    with open(f"{out_dir}/Yf.txt", "w") as f:
        for i in range(NUM_YF):
            f.write(("1_tok%d\n" % (i % NUM_XF)) if i % 3 == 0 else ("__label__%d__lbltok%d\n" % (i, i)))
    return out_dir


if __name__ == "__main__":
    print("wrote", make(sys.argv[1] if len(sys.argv) > 1 else "GZXML-Datasets/Synth"))
