"""End-to-end check of label feature-bag expansion *through* ``build_dataset``.

    python benchmarks/expansion_check.py GZXML-Datasets/GZ-Reuters-90 <glove.txt>
    python benchmarks/expansion_check.py GZXML-Datasets/GZ-NPM <glove.txt> shortyK=100 bs_count=40

The measured sweep that produced the numbers in the README rewrote ``Yf.txt`` and
``Y_Yf.txt`` of an already-built dataset, so it isolated the effect perfectly but never
exercised the ``label_expand`` hook on the path a user would actually take. This does:
it reads a dataset directory back into raw texts and label names, rebuilds it twice with
:func:`zestxml.dataset.build_dataset` -- once plain, once with
:func:`zestxml.embed.glove_expander` -- trains both, and prints the two evaluations side
by side.

Both arms are rebuilt here, so the control is the rebuild and not the shipped dataset:
the tf-idf vectorizer is refit, so absolute numbers shift a little and only the *delta*
between the two columns is meaningful.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zestxml import ZestXML, build_dataset  # noqa: E402
from zestxml.embed import glove_expander  # noqa: E402
from zestxml.io import read_desc_file, read_text_smat  # noqa: E402

# the reference configuration for GZ-Reuters-90 (Results/Reu2exact/params.txt)
CONFIG = dict(shortyK=50, bs_count=20, bs_alpha=0.02, bs_direct_wt=0.8,
              bilinear_classifier_cost=5, bilinear_normalize=0,
              bilinear_classifier_maxitr=20, num_thread=0)


def load(data_dir):
    """Recover (texts, label names) per split from a built dataset directory."""
    names = read_desc_file(f"{data_dir}/Y.txt")

    def rows(path):
        mat = read_text_smat(path)
        return [[names[i] for i in mat.indices[mat.indptr[r]:mat.indptr[r + 1]].tolist()]
                for r in range(mat.nrows)]

    unseen = set()
    if os.path.exists(f"{data_dir}/unseen_labels.txt"):
        with open(f"{data_dir}/unseen_labels.txt") as f:
            unseen = {int(l.split()[0]) for l in f if l.strip()}
    return (read_desc_file(f"{data_dir}/trn_X.txt"), rows(f"{data_dir}/trn_X_Y.txt"),
            read_desc_file(f"{data_dir}/tst_X.txt"), rows(f"{data_dir}/tst_X_Y.txt"),
            names, unseen)


def main(data_dir, vectors, topk=2, min_sim=0.7, **overrides):
    config = {**CONFIG, **overrides}
    tag_of = os.path.basename(data_dir.rstrip('/'))
    trn_x, trn_y, tst_x, tst_y, names, unseen = load(data_dir)
    print(f"{len(trn_x)} train / {len(tst_x)} test, {len(names)} labels, {len(unseen)} unseen\n")

    out = {}
    for arm, expander in (("control", None),
                          (f"expand k={topk} floor={min_sim}",
                           glove_expander(vectors, topk=topk, min_sim=min_sim))):
        tag = f"ExpCheck-{tag_of}-" + ("ctrl" if expander is None else "wide")
        print(f"--- {arm} " + "-" * 50)
        build_dataset(f"GZXML-Datasets/{tag}", trn_x, trn_y, tst_x, tst_y,
                      label_names=names, unseen=unseen, label_expand=expander)
        out[arm] = ZestXML(f"GZXML-Datasets/{tag}", f"Results/{tag}", **config).run()
        print()

    print("=" * 70)
    for metric in ("P@1", "PSP@1", "PSP@5"):
        for split in ("all labels", "unseen only", "seen only"):
            a, b = (out[k][split][metric] for k in out)
            print("%-12s %-8s control %6.2f   expanded %6.2f   delta %+6.2f"
                  % (split, metric, a, b, b - a))


if __name__ == "__main__":
    kw = dict(a.split("=", 1) for a in sys.argv[3:])
    main(sys.argv[1], sys.argv[2], **kw)
