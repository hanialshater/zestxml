"""Evaluate every recorded run for a dataset and print one comparison table.

    python benchmarks/results_table.py GZ-NPM
    python benchmarks/results_table.py GZ-Reuters-90 --md

Every row is recomputed from the score matrix on disk by :func:`zestxml.eval.report`, so
the table cannot drift from the artifacts the way a hand-maintained one does. Rows are
sorted by unseen-label P@1, which is what these datasets exist to measure. A run whose
matrix does not match the dataset shape is skipped rather than silently misreported.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zestxml.eval import report  # noqa: E402
from zestxml.io import read_bin_smat, read_text_smat  # noqa: E402

# run directory -> label, per dataset. Order here is only the fallback order.
RUNS = {
    "GZ-NPM": [
        ("Npm2exact", "ZestXML (reference)"),
        ("ExpCheck-GZ-NPM-ctrl", "ZestXML, rebuilt control"),
        ("ExpCheck-GZ-NPM-wide", "+ label expansion k=2/0.7"),
        ("Npm3glovefb", "+ GloVe direct map, fallback"),
        ("Npm3gloveaug", "+ GloVe direct map, augment"),
        ("NpmFTfb", "+ fastText direct map, fallback"),
        ("Npm2fuzzy", "+ char n-gram direct map"),
        ("label_graph", "+ label-graph propagation"),
        ("lowrank", "low-rank factorisation r=64"),
        ("splade", "SPLADE-style learned sparse"),
        ("classical", "kNN + BM25 hybrid"),
        ("classical-bm25", "BM25 only, no training"),
        ("classical-knn25", "kNN k=25"),
        ("classical-centroid", "tf-idf centroid"),
        ("ova_linear", "OVA linear (1-vs-all)"),
        ("dense_probe", "dense probe, Numberbatch"),
        # produced by benchmarks/colab/ZestXML_benchmarks.ipynb, on a GPU box
        ("renee", "Renee (end-to-end encoder)"),
        ("npm-dense", "dense probe, MiniLM-L6"),
        ("npm-hybrid", "ZestXML on a hybrid shortlist"),
        ("splade-both", "SPLADE, no label id, normalised"),
    ],
    "GZ-Reuters-90": [
        ("Reu2exact", "ZestXML (reference)"),
        ("ExpCheck-ctrl", "ZestXML, rebuilt control"),
        ("ExpCheck-wide", "+ label expansion k=2/0.7"),
        ("Reu3glovefb", "+ GloVe direct map, fallback"),
        ("Reu3gloveaug", "+ GloVe direct map, augment"),
        ("Refute2reu-nodense", "re-tuned alpha, standardised"),
        ("classicalR-bm25", "BM25 only, no training"),
        ("classicalR-hybrid", "kNN + BM25 hybrid"),
        ("classicalR-knn25", "kNN k=25"),
        ("classicalR-centroid", "tf-idf centroid"),
        ("ova_linear_reuters", "OVA linear (1-vs-all)"),
    ],
}

COLS = [("P@1", "all labels"), ("P@5", "all labels"), ("PSP@1", "all labels"),
        ("PSP@5", "all labels"), ("P@1", "unseen only"), ("P@1", "seen only")]
HEAD = ["P@1", "P@5", "PSP@1", "PSP@5", "unseen P@1", "seen P@1"]


def discover(dataset, results):
    """Every run directory holding a score matrix, known label first, dir name otherwise."""
    known = dict(RUNS[dataset])
    listed = [(r, known[r]) for r, _ in RUNS[dataset]]
    extra = sorted(d for d in os.listdir(results)
                   if d not in known and os.path.exists(f"{results}/{d}/score_mat.bin"))
    return listed + [(d, d) for d in extra]


def collect(dataset, results="Results", scan=False):
    data_dir = f"GZXML-Datasets/{dataset}"
    shape = read_text_smat(f"{data_dir}/tst_X_Y.txt").shape
    rows, skipped = [], []
    for run, label in (discover(dataset, results) if scan else RUNS[dataset]):
        path = f"{results}/{run}/score_mat.bin"
        if not os.path.exists(path):
            skipped.append((label, "no artifact"))
            continue
        if read_bin_smat(path).shape != shape:
            if not scan:  # scanning sees every dataset's runs; only complain about asked-for ones
                skipped.append((label, "shape mismatch"))
            continue
        m = report(path, data_dir, verbose=False)
        rows.append((label, run, [m[split][c] if split in m else float("nan") for c, split in COLS]))
    rows.sort(key=lambda r: -r[2][4])  # unseen P@1
    return rows, skipped


def main(dataset, markdown=False, results="Results", scan=False):
    rows, skipped = collect(dataset, results, scan)
    truth = read_text_smat(f"GZXML-Datasets/{dataset}/tst_X_Y.txt")
    unseen = sum(1 for _ in open(f"GZXML-Datasets/{dataset}/unseen_labels.txt"))
    print(f"# {dataset} -- {truth.nrows} test points, {truth.ncols} labels ({unseen} unseen)\n")

    if markdown:
        print("| method | " + " | ".join(HEAD) + " |")
        print("|---" * (len(HEAD) + 1) + "|")
        for label, _, vals in rows:
            print(f"| {label} | " + " | ".join("%.2f" % v for v in vals) + " |")
    else:
        print("%-34s " % "method" + " ".join("%11s" % h for h in HEAD))
        for label, _, vals in rows:
            print("%-34s " % label + " ".join("%11.2f" % v for v in vals))
    for label, why in skipped:
        print(f"\nskipped: {label} ({why})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=sorted(RUNS))
    ap.add_argument("--md", action="store_true")
    ap.add_argument("--results", default="Results")
    ap.add_argument("--scan", action="store_true",
                    help="also include run directories this script has no label for")
    a = ap.parse_args()
    main(a.dataset, a.md, a.results, a.scan)
