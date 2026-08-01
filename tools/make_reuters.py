"""Build a small *real* GZXML dataset from Reuters-21578, for validating the pipeline.

    pip install nltk scikit-learn
    python tools/make_reuters.py GZXML-Datasets/GZ-Reuters-90

Reuters-21578 (ModApte split) is genuinely multi-label and, more importantly for
generalized zero-shot XML, its topics are named in plain words ("grain",
"money-supply"), so labels have real text features that overlap the document
vocabulary. That is what ``Y_Yf`` needs.

The generalized zero-shot setting is created the way the paper's datasets are: a slice of
the topics is stripped out of ``trn_X_Y`` while staying in ``tst_X_Y`` and ``Y_Yf``, so
those labels have no training point and can only be reached through their own tokens.

Label features follow the convention documented in the README: token features are written
as ``1_<token>`` so that everything after the first underscore matches an entry of
``Xf.txt`` (this is what ``create_Xf_Yf_map_direct`` keys on), plus one unique
``__label__<i>__<name>`` feature per label that gives seen labels their own parameters.
"""

import os
import sys
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

UNSEEN_STRIDE = 3  # hold out every n-th eligible topic
UNSEEN_MIN_TEST = 8  # only hold out topics with enough test support to measure


def write_smat(path, rows, ncols):
    with open(path, "w") as f:
        f.write("%d %d\n" % (len(rows), ncols))
        for row in rows:
            f.write(" ".join("%d:%.5f" % (i, v) for i, v in row) + "\n")


def one_line(text):
    """Collapse every kind of whitespace to single spaces.

    Text goes into line-per-row files that are read back with universal newlines, so a
    stray \r or \v inside a description silently becomes an extra row and misaligns the
    text against the label matrix. str.split() with no argument splits on all of them.
    """
    return " ".join(str(text).split())


def write_lines(path, texts, expected):
    """Write one text per line, and refuse to emit a file that would misalign."""
    lines = [one_line(t) for t in texts]
    assert len(lines) == expected, f"{path}: {len(lines)} texts but {expected} rows expected"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(path, encoding="utf-8") as f:  # verify what a reader will actually see
        got = sum(1 for _ in f)
    assert got == expected, f"{path}: writes {expected} rows but reads back as {got}"


def csr_rows(mat):
    mat = mat.tocsr()
    mat.sort_indices()
    return [
        list(zip(mat.indices[mat.indptr[i] : mat.indptr[i + 1]].tolist(),
                 mat.data[mat.indptr[i] : mat.indptr[i + 1]].tolist()))
        for i in range(mat.shape[0])
    ]


def build(out_dir):
    from nltk.corpus import reuters

    os.makedirs(out_dir, exist_ok=True)
    labels = sorted(reuters.categories())
    label_id = {name: i for i, name in enumerate(labels)}

    trn_ids, tst_ids = [], []
    for fid in reuters.fileids():
        if not reuters.categories(fid):
            continue
        (trn_ids if fid.startswith("training/") else tst_ids).append(fid)

    trn_text = [reuters.raw(f) for f in trn_ids]
    tst_text = [reuters.raw(f) for f in tst_ids]

    # tf-idf is fit on the training split only
    vec = TfidfVectorizer(
        lowercase=True, token_pattern=r"[a-z]{2,}", ngram_range=(1, 2),
        min_df=3, sublinear_tf=True, stop_words="english",
    )
    trn_X_Xf = vec.fit_transform(trn_text)
    tst_X_Xf = vec.transform(tst_text)
    Xf = list(vec.get_feature_names_out())
    xf_set = set(Xf)

    # ---- label features -----------------------------------------------------
    Yf, yf_id = [], {}

    def feat(name):
        if name not in yf_id:
            yf_id[name] = len(Yf)
            Yf.append(name)
        return yf_id[name]

    Y_Yf_rows = [[] for _ in labels]
    has_text = [False] * len(labels)
    for i, name in enumerate(labels):
        toks = [t for t in name.replace("-", " ").split() if t]
        cells = {feat("__label__%d__%s" % (i, name)): 1.0}
        for t in toks:  # every token of the name becomes a label feature, matched or not
            cells[feat("1_" + t)] = 1.0
            has_text[i] = has_text[i] or t in xf_set
        phrase = " ".join(toks)
        if len(toks) > 1:
            cells[feat("1_" + phrase)] = 1.0
            has_text[i] = has_text[i] or phrase in xf_set
        Y_Yf_rows[i] = sorted(cells.items())
    linked = sum(has_text)

    # ---- label matrices -----------------------------------------------------
    def label_matrix(ids):
        return [sorted((label_id[c], 1.0) for c in reuters.categories(f)) for f in ids]

    trn_X_Y = label_matrix(trn_ids)
    tst_X_Y = label_matrix(tst_ids)

    trn_freq = Counter(i for row in trn_X_Y for i, _ in row)
    tst_freq = Counter(i for row in tst_X_Y for i, _ in row)

    # only hold out topics whose name reaches the document vocabulary: a topic with no
    # usable text is unreachable for any zero-shot method and would just dilute the metric
    eligible = [
        i for i in range(len(labels))
        if tst_freq[i] >= UNSEEN_MIN_TEST and trn_freq[i] > 0 and has_text[i]
    ]
    eligible.sort(key=lambda i: -tst_freq[i])
    unseen = set(eligible[1::UNSEEN_STRIDE])  # skip the single largest topic
    unseen |= {i for i in range(len(labels)) if trn_freq[i] == 0}  # naturally unseen ones

    trn_X_Y = [[(i, v) for i, v in row if i not in unseen] for row in trn_X_Y]

    # ---- write --------------------------------------------------------------
    # raw text, aligned row-for-row with the matrices below (see make_npm.py)
    write_lines(f"{out_dir}/trn_X.txt", trn_text, len(trn_ids))
    write_lines(f"{out_dir}/tst_X.txt", tst_text, len(tst_ids))
    write_lines(f"{out_dir}/Y.txt", [n.replace("-", " ") for n in labels], len(labels))
    for fname in ("trn_filter_labels.txt", "tst_filter_labels.txt"):
        open(f"{out_dir}/{fname}", "w").close()

    write_smat(f"{out_dir}/trn_X_Xf.txt", csr_rows(trn_X_Xf), len(Xf))
    write_smat(f"{out_dir}/tst_X_Xf.txt", csr_rows(tst_X_Xf), len(Xf))
    write_smat(f"{out_dir}/Y_Yf.txt", Y_Yf_rows, len(Yf))
    write_smat(f"{out_dir}/trn_X_Y.txt", trn_X_Y, len(labels))
    write_smat(f"{out_dir}/tst_X_Y.txt", tst_X_Y, len(labels))
    for name, vocab in (("Xf", Xf), ("Yf", Yf)):
        with open(f"{out_dir}/{name}.txt", "w") as f:
            f.write("\n".join(vocab) + "\n")
    with open(f"{out_dir}/unseen_labels.txt", "w") as f:
        for i in sorted(unseen):
            f.write("%d %s\n" % (i, labels[i]))

    unseen_test_mass = sum(tst_freq[i] for i in unseen)
    total_test_mass = sum(tst_freq.values())
    print(f"points      : {len(trn_ids)} train / {len(tst_ids)} test")
    print(f"labels      : {len(labels)} ({len(unseen)} unseen at train time)")
    print(f"features    : {len(Xf)} point / {len(Yf)} label")
    print(f"label text  : {linked}/{len(labels)} topics have a token in the point vocabulary")
    print(f"positives   : {sum(len(r) for r in trn_X_Y)} train / {sum(len(r) for r in tst_X_Y)} test")
    print(f"unseen mass : {unseen_test_mass}/{total_test_mass} test positives "
          f"({100.0 * unseen_test_mass / total_test_mass:.1f}%) belong to unseen labels")
    print("wrote", out_dir)


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "GZXML-Datasets/GZ-Reuters-90")
