"""Convert a dataset from the Extreme Classification Repository into GZXML format.

    python benchmarks/datasets/make_xmc.py raw/LF-AmazonTitles-131K \
        GZXML-Datasets/LF-AmazonTitles-131K --unseen_frac 0.1

The repository's "LF-" datasets ship raw text for both documents *and* labels, which is
what ZestXML needs -- a label with no training example is only reachable through the text
of its name. This reads that raw text and hands it to :func:`zestxml.dataset.build_dataset`,
so the tf-idf features, the label feature-bag conventions and the row-alignment checks are
the same ones every other dataset here goes through.

Layouts seen in the wild, both handled:

* ``trn.json.gz`` / ``tst.json.gz`` / ``lbl.json.gz`` -- one JSON object per line, with a
  ``title`` (and often ``content``) field, and ``target_ind`` giving the label ids.
* ``trn_X_Y.txt`` / ``tst_X_Y.txt`` plus ``train_raw_texts.txt`` /
  ``test_raw_texts.txt`` / ``label_raw_texts.txt`` -- one text per line.

``--unseen_frac`` holds out that fraction of labels from the *training* matrix to create a
generalized zero-shot split, the same construction the GZXML datasets use. Pass 0 to keep
every label seen and reproduce the standard (non zero-shot) benchmark, which is what the
published P@k numbers are measured on.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from zestxml.dataset import build_dataset, one_line  # noqa: E402


def _open(path):
    return gzip.open(path, "rt", encoding="utf-8") if path.endswith(".gz") else open(path, encoding="utf-8")


def read_json_split(path, use_content):
    texts, targets = [], []
    with _open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            title = row.get("title") or ""
            body = row.get("content") or "" if use_content else ""
            texts.append(one_line(f"{title} {body}"))
            targets.append(row.get("target_ind") or [])
    return texts, targets


def read_lines(path):
    with _open(path) as f:
        return [one_line(l) for l in f]


def read_smat_rows(path):
    """The repository's X_Y matrices: a `nrows ncols` header then `idx:val` lines."""
    rows = []
    with open(path) as f:
        f.readline()
        for line in f:
            rows.append([int(t.split(":")[0]) for t in line.split()])
    return rows


def load(src, use_content):
    """Return (trn_texts, trn_ids, tst_texts, tst_ids, label_names) for either layout."""
    j = os.path.join
    if os.path.exists(j(src, "trn.json.gz")) or os.path.exists(j(src, "trn.json")):
        ext = ".json.gz" if os.path.exists(j(src, "trn.json.gz")) else ".json"
        trn_texts, trn_ids = read_json_split(j(src, "trn" + ext), use_content)
        tst_texts, tst_ids = read_json_split(j(src, "tst" + ext), use_content)
        lbl_texts, _ = read_json_split(j(src, "lbl" + ext), use_content)
        return trn_texts, trn_ids, tst_texts, tst_ids, lbl_texts

    need = ["trn_X_Y.txt", "tst_X_Y.txt", "train_raw_texts.txt", "test_raw_texts.txt",
            "label_raw_texts.txt"]
    missing = [n for n in need if not os.path.exists(j(src, n))]
    if missing:
        raise SystemExit(
            f"{src}: recognised neither layout.\n"
            f"  json layout needs trn/tst/lbl.json[.gz]\n"
            f"  text layout needs {', '.join(need)} (missing {', '.join(missing)})\n"
            f"  present: {', '.join(sorted(os.listdir(src))[:20])}")
    return (read_lines(j(src, "train_raw_texts.txt")), read_smat_rows(j(src, "trn_X_Y.txt")),
            read_lines(j(src, "test_raw_texts.txt")), read_smat_rows(j(src, "tst_X_Y.txt")),
            read_lines(j(src, "label_raw_texts.txt")))


def main(src, out, unseen_frac=0.0, use_content=False, min_df=3, seed=0):
    trn_texts, trn_ids, tst_texts, tst_ids, names = load(src, use_content)
    print(f"read {len(trn_texts)} train / {len(tst_texts)} test / {len(names)} labels")
    assert len(trn_texts) == len(trn_ids), f"{len(trn_texts)} texts vs {len(trn_ids)} label rows"
    assert len(tst_texts) == len(tst_ids), f"{len(tst_texts)} texts vs {len(tst_ids)} label rows"

    # build_dataset takes label NAMES per document, so map ids through the label list
    def as_names(rows):
        return [[names[i] for i in row if 0 <= i < len(names)] for row in rows]

    unseen = None
    if unseen_frac > 0:
        from zestxml.dataset import select_unseen_labels
        stride = max(2, int(round(1.0 / unseen_frac)))
        unseen = select_unseen_labels(trn_ids, tst_ids, len(names), stride=stride, min_test=1)
        print(f"holding {len(unseen)} of {len(names)} labels out of training "
              f"({100.0 * len(unseen) / len(names):.1f}%)")

    from sklearn.feature_extraction.text import TfidfVectorizer
    return build_dataset(
        out, trn_texts, as_names(trn_ids), tst_texts, as_names(tst_ids),
        label_names=names, unseen=unseen,
        vectorizer=TfidfVectorizer(lowercase=True, token_pattern=r"[a-z][a-z0-9]+",
                                   ngram_range=(1, 2), min_df=min_df, sublinear_tf=True,
                                   stop_words="english"),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="directory holding the repository's raw files")
    ap.add_argument("out", help="GZXML dataset directory to write")
    ap.add_argument("--unseen_frac", type=float, default=0.0,
                    help="fraction of labels to hide from training; 0 reproduces the "
                         "standard benchmark that published P@k numbers are measured on")
    ap.add_argument("--content", action="store_true",
                    help="append the 'content' field to the title (the -Titles- variants "
                         "of these datasets are title-only by definition, so leave this off "
                         "when reproducing their numbers)")
    ap.add_argument("--min_df", type=int, default=3)
    a = ap.parse_args()
    main(a.src, a.out, a.unseen_frac, a.content, a.min_df)
