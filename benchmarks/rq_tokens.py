"""Rewrite a dataset directory with RQ-KMeans "semantic ID" codes as features.

    python benchmarks/rq_tokens.py GZXML-Datasets/GZ-Reuters-90 \
        GZXML-Datasets/Rq0-reu-replace-L4K64 --mode replace --levels 4 -K 64 \
        --vectors /path/glove100.gz

Two arms:

``replace``  the rq codes stand *instead of* the lexical tokens.  A document row is
             exactly its L codes; a label row is its unique ``__label__<i>__<name>``
             feature plus its L ``1_rq<l>_<c>`` codes.
``augment``  the rq codes sit *alongside* the lexical features.

Only ``Xf.txt``, ``Yf.txt``, ``trn_X_Xf.txt``, ``tst_X_Xf.txt`` and ``Y_Yf.txt`` are
rewritten.  ``trn_X_Y.txt``, ``tst_X_Y.txt``, ``unseen_labels.txt``, ``trn_X.txt``,
``tst_X.txt``, ``Y.txt`` and the filter files are **hard-linked** from the source, so the
split and the unseen label set are byte-identical to the control by construction.

The codebook is fit on the TRAINING DOCUMENTS ONLY.  Test documents and label names are
quantized with those already-fitted centroids.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zestxml.embed import load_word_vectors
from zestxml.io import read_desc_file
from zestxml.quantize import embed_texts, rq_kmeans, sharing_report, simple_tokens

LINK = ["trn_X_Y.txt", "tst_X_Y.txt", "unseen_labels.txt", "trn_X.txt", "tst_X.txt",
        "Y.txt", "trn_filter_labels.txt", "tst_filter_labels.txt", "pos_trn_tst.txt"]


def read_rows(path):
    """Read a text sparse matrix as (nrows, ncols, [[(j, v), ...], ...])."""
    with open(path) as f:
        head = f.readline().split()
        nr, nc = int(head[0]), int(head[1])
        rows = []
        for _ in range(nr):
            line = f.readline().split()
            rows.append([(int(a), float(b)) for a, b in (e.split(":") for e in line)])
    return nr, nc, rows


def write_rows(path, rows, ncols):
    with open(path, "w") as f:
        f.write("%d %d\n" % (len(rows), ncols))
        for r in rows:
            f.write(" ".join("%d:%.5f" % (j, v) for j, v in sorted(r)) + "\n")


def hardlink(src_dir, dst_dir, names=LINK):
    for name in names:
        s, d = os.path.join(src_dir, name), os.path.join(dst_dir, name)
        if os.path.exists(s) and not os.path.exists(d):
            os.link(s, d)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--mode", choices=["replace", "augment"], default="replace")
    ap.add_argument("--levels", "-L", type=int, default=4)
    ap.add_argument("-K", "--codebook", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vectors", required=True)
    ap.add_argument("--weight", type=float, default=1.0,
                    help="value of an rq feature. Rows are unit-normalised downstream, so "
                         "in replace mode this is irrelevant (L equal entries); in augment "
                         "mode it sets how much the rq block dilutes the lexical block.")
    ap.add_argument("--doc_max_tokens", type=int, default=200)
    ap.add_argument("--examples", type=int, default=6)
    args = ap.parse_args(argv)

    src, dst = args.src, args.dst
    os.makedirs(dst, exist_ok=True)

    Xf: List[str] = read_desc_file(f"{src}/Xf.txt")
    Yf: List[str] = read_desc_file(f"{src}/Yf.txt")
    Y: List[str] = read_desc_file(f"{src}/Y.txt")
    trn_text = read_desc_file(f"{src}/trn_X.txt")
    tst_text = read_desc_file(f"{src}/tst_X.txt")
    ntr, nxf, trn_rows = read_rows(f"{src}/trn_X_Xf.txt")
    nte, _, tst_rows = read_rows(f"{src}/tst_X_Xf.txt")
    nlb, nyf, y_rows = read_rows(f"{src}/Y_Yf.txt")
    assert len(trn_text) == ntr and len(tst_text) == nte and len(Y) == nlb

    # ---- vectors, restricted to the tokens this dataset can possibly ask for -------
    need = set()
    for t in trn_text[:]:
        need.update(simple_tokens(t)[: args.doc_max_tokens])
    for t in tst_text:
        need.update(simple_tokens(t)[: args.doc_max_tokens])
    for n in Y:
        need.update(simple_tokens(n))
    index, vecs = load_word_vectors(args.vectors, vocab=need)
    print("vectors: %d/%d needed tokens have one" % (len(index), len(need)))

    # ---- embed. names require FULL coverage; documents average over known tokens ----
    lab_emb, lab_keep = embed_texts(Y, index, vecs, require_all=True)
    trn_emb, trn_keep = embed_texts(trn_text, index, vecs, require_all=False,
                                    max_tokens=args.doc_max_tokens)
    tst_emb, tst_keep = embed_texts(tst_text, index, vecs, require_all=False,
                                    max_tokens=args.doc_max_tokens)
    print("embeddable: %d/%d train docs, %d/%d test docs, %d/%d labels "
          "(labels need every token known)"
          % (len(trn_keep), ntr, len(tst_keep), nte, len(lab_keep), nlb))

    # ---- fit RQ on TRAINING DOCUMENTS ONLY -----------------------------------------
    print("\nfitting RQ-KMeans on %d training documents (L=%d, K=%d, seed=%d)"
          % (len(trn_keep), args.levels, args.codebook, args.seed))
    cb, trn_codes = rq_kmeans(trn_emb, args.levels, args.codebook, seed=args.seed)
    tst_codes = cb.transform(tst_emb)
    lab_codes = cb.transform(lab_emb)

    print("\nsharing over LABELS (this is the diagnostic that predicts the sign):")
    for line in sharing_report(lab_codes):
        print("  " + line)
    print("sharing over TRAIN DOCUMENTS:")
    for line in sharing_report(trn_codes):
        print("  " + line)

    # ---- new vocabularies ------------------------------------------------------------
    used = sorted({(l, int(c)) for codes in (trn_codes, tst_codes, lab_codes)
                   for l in range(args.levels) for c in codes[:, l]})
    rq_names = ["rq%d_%d" % (l, c) for l, c in used]
    rq_xf = {}
    rq_yf = {}
    if args.mode == "replace":
        new_Xf = list(rq_names)
        for i, n in enumerate(rq_names):
            rq_xf[n] = i
        new_Yf = [n for n in Yf if n.startswith("__label__")]
        keep_yf = {n: i for i, n in enumerate(new_Yf)}
        old_to_new_yf = {j: keep_yf[n] for j, n in enumerate(Yf) if n in keep_yf}
        for n in rq_names:
            rq_yf["1_" + n] = len(new_Yf)
            new_Yf.append("1_" + n)
    else:
        new_Xf = list(Xf)
        for n in rq_names:
            rq_xf[n] = len(new_Xf)
            new_Xf.append(n)
        new_Yf = list(Yf)
        old_to_new_yf = {j: j for j in range(len(Yf))}
        for n in rq_names:
            rq_yf["1_" + n] = len(new_Yf)
            new_Yf.append("1_" + n)

    w = args.weight

    def doc_rows(base_rows, codes, keep, n):
        code_of = {int(i): codes[r] for r, i in enumerate(keep)}
        out = []
        empty = 0
        for i in range(n):
            row = [] if args.mode == "replace" else list(base_rows[i])
            c = code_of.get(i)
            if c is not None:
                for l in range(args.levels):
                    row.append((rq_xf["rq%d_%d" % (l, int(c[l]))], w))
            if not row:
                empty += 1
            out.append(row)
        return out, empty

    new_trn, e_trn = doc_rows(trn_rows, trn_codes, trn_keep, ntr)
    new_tst, e_tst = doc_rows(tst_rows, tst_codes, tst_keep, nte)

    lab_code_of = {int(i): lab_codes[r] for r, i in enumerate(lab_keep)}
    new_y, e_lab = [], 0
    for i in range(nlb):
        row = [(old_to_new_yf[j], v) for j, v in y_rows[i] if j in old_to_new_yf]
        c = lab_code_of.get(i)
        if c is not None:
            for l in range(args.levels):
                row.append((rq_yf["1_rq%d_%d" % (l, int(c[l]))], w))
        else:
            e_lab += 1
        new_y.append(row)
    print("\nno rq code at all: %d train docs, %d test docs, %d labels (OOV)"
          % (e_trn, e_tst, e_lab))

    write_rows(f"{dst}/trn_X_Xf.txt", new_trn, len(new_Xf))
    write_rows(f"{dst}/tst_X_Xf.txt", new_tst, len(new_Xf))
    write_rows(f"{dst}/Y_Yf.txt", new_y, len(new_Yf))
    with open(f"{dst}/Xf.txt", "w") as f:
        f.write("\n".join(new_Xf) + "\n")
    with open(f"{dst}/Yf.txt", "w") as f:
        f.write("\n".join(new_Yf) + "\n")
    hardlink(src, dst)
    print("wrote %s  (num_Xf %d -> %d, num_Yf %d -> %d)"
          % (dst, nxf, len(new_Xf), nyf, len(new_Yf)))

    # ---- sanity: unseen labels and their nearest training documents by shared code ---
    unseen = []
    up = f"{src}/unseen_labels.txt"
    if os.path.exists(up):
        unseen = [int(l.split()[0]) for l in open(up) if l.strip()]
    print("\n--- sanity: unseen labels and the training docs sharing most rq codes ---")
    tr_index = {int(i): r for r, i in enumerate(trn_keep)}
    for lid in unseen[: args.examples]:
        if lid not in lab_code_of:
            print("%-16s : OOV, no codes" % Y[lid])
            continue
        c = lab_code_of[lid]
        share = (trn_codes == c[None, :]).sum(axis=1)
        order = np.argsort(-share)[:3]
        print("%-16s codes=%s" % (Y[lid], list(map(int, c))))
        for o in order:
            doc = int(trn_keep[o])
            print("    share=%d  %s" % (int(share[o]), trn_text[doc][:110].replace("\n", " ")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
