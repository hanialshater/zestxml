"""Turn raw texts and labels into a GZXML dataset directory.

    from zestxml.dataset import build_dataset

    build_dataset(
        "GZXML-Datasets/MyData",
        trn_texts=["hiking boots waterproof", ...],
        trn_labels=[["hiking", "outdoors"], ...],
        tst_texts=[...],
        tst_labels=[...],
    )

That writes every file `run_torch.py`, `tools/eval_xc.py` and encoder baselines such as
Renee expect. The point of doing it here rather than by hand is the conventions, which are
not obvious and are easy to get wrong:

* **Label features carry the zero-shot mechanism.** Each label becomes a bag of features:
  ``1_<token>`` for every token of its name, ``1_<whole phrase>`` when it has several, and
  one unique ``__label__<i>__<name>``. The ``1_`` prefix matters -- the direct map strips
  everything up to the first underscore and looks the rest up in the point vocabulary, so
  ``1_hiking`` links to the document feature ``hiking``. That link is how a label with no
  training example is reachable at all.
* **Emit every token, matched or not.** Only emitting tokens already in the document
  vocabulary makes every label trivially matchable and quietly removes the population that
  fuzzy matching exists to serve.
* **Text files must stay row-aligned with the matrices.** Descriptions containing
  ``\\r`` silently become extra rows when read back, misaligning text against labels;
  Renee in particular maps line N to row N and never checks. Every write is verified here.
* **tf-idf is fit on training text only.**
"""

from __future__ import annotations

import os
import re
from collections import Counter
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set

DEFAULT_TOKEN = re.compile(r"[a-z][a-z0-9]+")


def default_label_tokens(name: str) -> List[str]:
    return DEFAULT_TOKEN.findall(name.lower())


def one_line(text) -> str:
    """Collapse every kind of whitespace, so one record stays one line."""
    return " ".join(str(text).split())


def write_lines(path: str, texts: Iterable[str], expected: int) -> None:
    """Write one text per line, refusing to emit a file that would misalign."""
    lines = [one_line(t) for t in texts]
    assert len(lines) == expected, f"{path}: {len(lines)} texts but {expected} rows expected"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(path, encoding="utf-8") as f:  # what a reader will actually see
        got = sum(1 for _ in f)
    assert got == expected, f"{path}: wrote {expected} rows but reads back as {got}"


def write_smat(path: str, rows, ncols: int) -> None:
    with open(path, "w") as f:
        f.write("%d %d\n" % (len(rows), ncols))
        for row in rows:
            f.write(" ".join("%d:%.5f" % (i, v) for i, v in row) + "\n")


def csr_rows(mat) -> List[List]:
    mat = mat.tocsr()
    mat.sort_indices()
    return [
        list(zip(mat.indices[mat.indptr[i] : mat.indptr[i + 1]].tolist(),
                 mat.data[mat.indptr[i] : mat.indptr[i + 1]].tolist()))
        for i in range(mat.shape[0])
    ]


def select_unseen_labels(
    trn_labels: Sequence[Sequence[int]], tst_labels: Sequence[Sequence[int]], n_labels: int,
    stride: int = 4, min_test: int = 8, reachable: Optional[Sequence[bool]] = None,
) -> Set[int]:
    """Choose labels to hold out of training, for a generalized zero-shot split.

    Every ``stride``-th eligible label, plus any label with no training positive anyway.
    A label needs enough test support to be measurable, and -- when ``reachable`` is given
    -- text that reaches the document vocabulary, since a label no method could ever
    retrieve only dilutes the metric.
    """
    trn_freq = Counter(i for row in trn_labels for i in row)
    tst_freq = Counter(i for row in tst_labels for i in row)
    eligible = [
        i for i in range(n_labels)
        if tst_freq[i] >= min_test and trn_freq[i] > 0 and (reachable is None or reachable[i])
    ]
    eligible.sort(key=lambda i: -tst_freq[i])
    unseen = set(eligible[1::stride])  # skip the single largest label
    unseen |= {i for i in range(n_labels) if trn_freq[i] == 0}
    return unseen


def build_dataset(
    out_dir: str,
    trn_texts: Sequence[str],
    trn_labels: Sequence[Sequence[str]],
    tst_texts: Sequence[str],
    tst_labels: Sequence[Sequence[str]],
    label_names: Optional[Sequence[str]] = None,
    unseen: Optional[Set[int]] = None,
    label_tokens: Callable[[str], List[str]] = default_label_tokens,
    label_expand=None,
    label_text: Optional[Callable[[str], str]] = None,
    vectorizer=None,
    verbose: bool = True,
) -> Dict:
    """Write a GZXML dataset directory. Labels are given as names per document.

    ``unseen`` (label ids) are stripped from the training matrix only: they keep their
    features and their test positives, which is what makes them zero-shot rather than
    absent. Pass :func:`select_unseen_labels` if you want a benchmark-style split, or
    ``None`` for an ordinary dataset.

    ``label_expand(names, name_tokens, vocab) -> {name: extra tokens}`` optionally widens
    each label's feature bag beyond the words of its own name -- see
    :func:`zestxml.embed.glove_expander`. It is called once with the whole label set and
    the point vocabulary. Default ``None`` leaves the output byte-identical to a build
    without the hook. Read the README before switching it on: it is worth several points
    of unseen accuracy when the vector space is in-domain for the label names, and costs
    as many when it is not.

    Returns a dict of statistics, also printed when ``verbose``.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    os.makedirs(out_dir, exist_ok=True)
    assert len(trn_texts) == len(trn_labels), "train texts and labels differ in length"
    assert len(tst_texts) == len(tst_labels), "test texts and labels differ in length"

    if label_names is None:
        label_names = sorted({l for row in list(trn_labels) + list(tst_labels) for l in row})
    label_id = {name: i for i, name in enumerate(label_names)}
    n_labels = len(label_names)

    if vectorizer is None:
        vectorizer = TfidfVectorizer(
            lowercase=True, token_pattern=r"[a-z][a-z0-9]+", ngram_range=(1, 2),
            min_df=3, sublinear_tf=True, stop_words="english",
        )
    trn_X_Xf = vectorizer.fit_transform(trn_texts)  # fit on training text only
    tst_X_Xf = vectorizer.transform(tst_texts)
    Xf = list(vectorizer.get_feature_names_out())
    xf_set = set(Xf)

    # ---- label features -----------------------------------------------------
    Yf: List[str] = []
    yf_id: Dict[str, int] = {}

    def feat(name: str) -> int:
        if name not in yf_id:
            yf_id[name] = len(Yf)
            Yf.append(name)
        return yf_id[name]

    tokens_of = [label_tokens(name) for name in label_names]
    extra = label_expand(label_names, tokens_of, Xf) if label_expand is not None else {}

    Y_Yf_rows, reachable = [], []
    expanded = Counter()
    for i, name in enumerate(label_names):
        cells = {feat("__label__%d__%s" % (i, name)): 1.0}
        toks = tokens_of[i]
        hit = False
        for t in toks:  # every token, in the vocabulary or not
            cells[feat("1_" + t)] = 1.0
            hit = hit or t in xf_set
        phrase = " ".join(toks)
        if len(toks) > 1:
            cells[feat("1_" + phrase)] = 1.0
            hit = hit or phrase in xf_set
        # Extra tokens the label does not literally contain, emitted in the same "1_"
        # namespace as its own, so an unseen label can borrow the *trained* weight of a
        # feature that seen labels already carry rather than only the untrained knn
        # channel. They deliberately do not count towards ``reachable``, which stays a
        # property of the label's own text.
        for t in extra.get(name, ()):
            cell = feat("1_" + t)
            if cell not in cells:
                expanded[t] += 1
            cells[cell] = 1.0
        Y_Yf_rows.append(sorted(cells.items()))
        reachable.append(hit)

    # ---- label matrices -----------------------------------------------------
    def as_ids(rows):
        return [sorted({label_id[l] for l in row if l in label_id}) for row in rows]

    trn_ids, tst_ids = as_ids(trn_labels), as_ids(tst_labels)
    unseen = set() if unseen is None else set(unseen)
    trn_matrix = [[(i, 1.0) for i in row if i not in unseen] for row in trn_ids]
    tst_matrix = [[(i, 1.0) for i in row] for row in tst_ids]

    # ---- write --------------------------------------------------------------
    write_lines(f"{out_dir}/trn_X.txt", trn_texts, len(trn_texts))
    write_lines(f"{out_dir}/tst_X.txt", tst_texts, len(tst_texts))
    write_lines(f"{out_dir}/Y.txt",
                [label_text(n) if label_text else n for n in label_names], n_labels)
    for fname in ("trn_filter_labels.txt", "tst_filter_labels.txt"):
        open(f"{out_dir}/{fname}", "w").close()

    write_smat(f"{out_dir}/trn_X_Xf.txt", csr_rows(trn_X_Xf), len(Xf))
    write_smat(f"{out_dir}/tst_X_Xf.txt", csr_rows(tst_X_Xf), len(Xf))
    write_smat(f"{out_dir}/Y_Yf.txt", Y_Yf_rows, len(Yf))
    write_smat(f"{out_dir}/trn_X_Y.txt", trn_matrix, n_labels)
    write_smat(f"{out_dir}/tst_X_Y.txt", tst_matrix, n_labels)
    for name, vocab in (("Xf", Xf), ("Yf", Yf)):
        with open(f"{out_dir}/{name}.txt", "w") as f:
            f.write("\n".join(vocab) + "\n")
    with open(f"{out_dir}/unseen_labels.txt", "w") as f:
        for i in sorted(unseen):
            f.write("%d %s\n" % (i, label_names[i]))

    tst_freq = Counter(i for row in tst_ids for i in row)
    stats = {
        "train_points": len(trn_texts), "test_points": len(tst_texts),
        "labels": n_labels, "unseen_labels": len(unseen),
        "point_features": len(Xf), "label_features": len(Yf),
        "labels_with_reachable_text": sum(reachable),
        "expanded_features": len(expanded),
        "labels_per_expanded_feature": (sum(expanded.values()) / len(expanded)) if expanded else 0.0,
        "max_labels_per_expanded_feature": max(expanded.values(), default=0),
        "train_positives": sum(len(r) for r in trn_matrix),
        "test_positives": sum(len(r) for r in tst_matrix),
        "unseen_test_positives": sum(tst_freq[i] for i in unseen),
        "out_dir": out_dir,
    }
    if verbose:
        print(f"points      : {stats['train_points']} train / {stats['test_points']} test")
        print(f"labels      : {n_labels} ({len(unseen)} unseen at train time)")
        print(f"features    : {len(Xf)} point / {len(Yf)} label")
        print(f"label text  : {sum(reachable)}/{n_labels} labels have a token in the point vocabulary")
        print(f"positives   : {stats['train_positives']} train / {stats['test_positives']} test")
        if expanded:
            # How widely an added feature is shared is what decides whether expansion
            # helps: a feature on 2 labels discriminates, one on 19 is noise. Reuters
            # measured 1.26 labels per added feature and gained; npm 1.54 with a tail to
            # 19 and lost. Watch this number, not the token coverage.
            top = ", ".join("%s x%d" % kv for kv in expanded.most_common(5))
            print(f"expansion   : {len(expanded)} added features, "
                  f"{stats['labels_per_expanded_feature']:.2f} labels each on average "
                  f"(max {stats['max_labels_per_expanded_feature']}); most shared: {top}")
        if unseen:
            share = 100.0 * stats["unseen_test_positives"] / max(1, stats["test_positives"])
            print(f"unseen mass : {stats['unseen_test_positives']}/{stats['test_positives']} "
                  f"test positives ({share:.1f}%) belong to unseen labels")
        print("wrote", out_dir)
    return stats
