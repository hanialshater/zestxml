"""Concatenate rq codes with lexical tokens as *separate blocks*, instead of weighting them.

    python benchmarks/rq_concat.py GZXML-Datasets/GZ-Reuters-90 GZXML-Datasets/Cat-reu \
        --vectors glove100.gz --levels 4 -K 64 --mode blocknorm

``benchmarks/rq_tokens.py --mode augment`` puts rq codes and lexical tokens in one row and
lets the pipeline L2-normalise the whole thing (``csr.py``: ``unit_normalize_rows``). That
makes the two blocks compete inside a single unit budget: adding L rq entries necessarily
shrinks every lexical entry, and the ``--weight`` knob only re-tunes the split. Whatever
weight you pick, a document's lexical representation changes just because codes were added.

So this script always normalises the two blocks *independently* and then gives each an
explicit share of the row. The lexical part is then identical to the control however many
codes are added, and the row arrives unit-norm so the pipeline's own normalisation is a
no-op. Two knobs, deliberately separate:

``--block_split``
    The *mass* of the rq block, as a share of row energy. This turns out to be the variable
    that decides everything -- weighting and concatenation are not really alternatives, they
    are two ways of setting this, one implicit and one exact.

``--mode``
    The *shape* of the rq block. ``blocknorm`` gives every code equal weight; ``idf`` weights
    each code by its inverse document frequency over the training documents, so a code
    sitting on many items counts for less -- learned from the data rather than chosen.

Keeping mass and shape apart is not fussiness. Raw idf weights average ~4.9 against
tf-idf's ~0.10, so concatenating them unscaled hands the rq block 98.9% of the row and
silently turns an "augment" arm into a "replace" arm. That mistake cost a whole measured
arm here before it was caught.

Only Xf.txt, Yf.txt, trn_X_Xf.txt, tst_X_Xf.txt and Y_Yf.txt are rewritten; trn_X_Y.txt,
tst_X_Y.txt, unseen_labels.txt and the raw text files are hard-linked from the source, so
the split and the unseen label set are byte-identical to the control by construction.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zestxml.dataset import write_smat  # noqa: E402
from zestxml.io import read_desc_file, read_text_smat  # noqa: E402
from zestxml.quantize import embed_texts, rq_kmeans, sharing_report  # noqa: E402
from zestxml.embed import load_word_vectors  # noqa: E402

LINKED = ["trn_X_Y.txt", "tst_X_Y.txt", "unseen_labels.txt", "trn_X.txt", "tst_X.txt",
          "Y.txt", "trn_filter_labels.txt", "tst_filter_labels.txt"]


def rows_of(mat):
    return [list(zip(mat.indices[mat.indptr[r]:mat.indptr[r + 1]].tolist(),
                     mat.values[mat.indptr[r]:mat.indptr[r + 1]].tolist()))
            for r in range(mat.nrows)]


def unit(row):
    n = math.sqrt(sum(v * v for _, v in row))
    return [(i, v / n) for i, v in row] if n > 0 else row


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--vectors", help="word2vec/GloVe text file (mean-of-token embedding)")
    ap.add_argument("--model", help="sentence-transformers model instead of --vectors, e.g. "
                                    "all-MiniLM-L6-v2, or clip-ViT-B-32 for a space shared "
                                    "with images. One encoder for both sides, so label names "
                                    "and documents land in the same region.")
    ap.add_argument("--mode", choices=["blocknorm", "idf"], default="blocknorm")
    ap.add_argument("--levels", "-L", type=int, default=4)
    ap.add_argument("-K", "--codebook", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--block_split", type=float, default=0.5,
                    help="blocknorm only: share of the row's mass given to the rq block")
    ap.add_argument("--doc_max_tokens", type=int, default=200)
    ap.add_argument("--label_codes", type=int, default=1,
                    help="assign the m nearest centroids per level to each LABEL instead of "
                         "just the nearest. Keeps every code as frequent as before -- so it "
                         "still fires, which is where --tuples failed -- while making the set "
                         "of codes specific: on synthetic data 398/400 items get a unique "
                         "code-set at m=3 against 288/400 at m=1.")
    ap.add_argument("--doc_codes", type=int, default=1, help="same, for documents")
    ap.add_argument("--tuples", action="store_true",
                    help="emit coarse-to-fine PREFIX TUPLES (rq0_a, rq01_a-b, rq012_a-b-c, ...) "
                         "instead of L independent marginals. The identity of an item lives in "
                         "the conjunction of its codes -- on GZ-Reuters-90 the 87 coded labels "
                         "occupy 18 distinct level-0 cells but 80-85 distinct full tuples -- so "
                         "emitting only marginals throws away what residual quantization built.")
    args = ap.parse_args(argv)

    src, dst = args.src, args.dst
    os.makedirs(dst, exist_ok=True)

    Xf, Yf = read_desc_file(f"{src}/Xf.txt"), read_desc_file(f"{src}/Yf.txt")
    trn_rows = rows_of(read_text_smat(f"{src}/trn_X_Xf.txt"))
    tst_rows = rows_of(read_text_smat(f"{src}/tst_X_Xf.txt"))
    y_rows = rows_of(read_text_smat(f"{src}/Y_Yf.txt"))
    trn_txt = read_desc_file(f"{src}/trn_X.txt")
    tst_txt = read_desc_file(f"{src}/tst_X.txt")
    names = read_desc_file(f"{src}/Y.txt")

    def clip(t):
        return " ".join(t.split()[: args.doc_max_tokens])

    if bool(args.model) == bool(args.vectors):
        ap.error("pass exactly one of --vectors (GloVe) or --model (sentence-transformers)")
    if args.model:
        from zestxml.quantize import encode_texts
        trn_emb, trn_keep = encode_texts([clip(t) for t in trn_txt], args.model)
        tst_emb, tst_keep = encode_texts([clip(t) for t in tst_txt], args.model)
        lab_emb, lab_keep = encode_texts(names, args.model)
    else:
        import torch
        index, table = load_word_vectors(args.vectors, torch.float32)
        # documents average over their known tokens; a name is embedded only when every
        # token is known, the rule that stops "middleware webpack" collapsing onto
        # "middleware". The two rules differ, which is the asymmetry --model removes.
        trn_emb, trn_keep = embed_texts([clip(t) for t in trn_txt], index, table, require_all=False)
        tst_emb, tst_keep = embed_texts([clip(t) for t in tst_txt], index, table, require_all=False)
        lab_emb, lab_keep = embed_texts(names, index, table, require_all=True)
    print("embedded %d/%d train, %d/%d test, %d/%d labels"
          % (len(trn_keep), len(trn_txt), len(tst_keep), len(tst_txt), len(lab_keep), len(names)))

    book, _ = rq_kmeans(trn_emb, args.levels, args.codebook, seed=args.seed)  # TRAIN ONLY
    trn_codes = book.transform_multi(trn_emb, args.doc_codes)
    tst_codes = book.transform_multi(tst_emb, args.doc_codes)
    lab_codes = book.transform_multi(lab_emb, args.label_codes)
    print(f"codebook fit on {len(trn_emb)} training documents only")
    for line in sharing_report(lab_codes[:, :, 0]):
        print("  labels: " + line)

    def feats_of(code):
        """Feature names for one item. ``code`` is (levels, m); column 0 is the nearest."""
        if not args.tuples:
            return sorted({"rq%d_%d" % (l, int(c)) for l in range(args.levels) for c in code[l]})
        return ["rq%s_%s" % ("".join(str(j) for j in range(l + 1)),
                             "-".join(str(int(code[j][0])) for j in range(l + 1)))
                for l in range(args.levels)]

    if args.label_codes > 1 or args.doc_codes > 1:
        sets = {tuple(feats_of(lab_codes[r])) for r in range(lab_codes.shape[0])}
        print("  labels: %d distinct code-sets over %d coded labels (label m=%d, doc m=%d)"
              % (len(sets), lab_codes.shape[0], args.label_codes, args.doc_codes))

    rq_names = sorted({n for codes in (trn_codes, tst_codes, lab_codes)
                       for r in range(codes.shape[0]) for n in feats_of(codes[r])})
    new_Xf = list(Xf) + rq_names
    new_Yf = list(Yf) + ["1_" + n for n in rq_names]
    xf_id = {n: len(Xf) + i for i, n in enumerate(rq_names)}
    yf_id = {n: len(Yf) + i for i, n in enumerate(rq_names)}

    # ---- rq feature weights ----------------------------------------------------------
    if args.mode == "idf":
        df = Counter()
        for r in range(trn_codes.shape[0]):
            for n in feats_of(trn_codes[r]):
                df[n] += 1
        n_doc = max(1, len(trn_codes))
        weight = {n: math.log((1.0 + n_doc) / (1.0 + df.get(n, 0))) + 1.0 for n in rq_names}
        top = sorted(weight.items(), key=lambda kv: kv[1])[:4]
        print("idf: lowest-weight codes (most shared) " + ", ".join("%s %.2f" % kv for kv in top))
    else:
        weight = {n: 1.0 for n in rq_names}

    rq_share = args.block_split
    lex_scale, rq_scale = math.sqrt(1 - rq_share), math.sqrt(rq_share)

    def combine(lexical, codes, prefix_id):
        rq = [(prefix_id[n], weight[n]) for n in feats_of(codes)] if codes is not None else []
        # Both blocks are normalised on their own and then given an explicit share of the
        # row, so the lexical part is unchanged by the presence or number of rq codes and
        # --block_split means exactly what it says. The mode chooses the *shape* of the rq
        # block (flat, or idf so a widely-shared code counts for less); --block_split
        # chooses its *mass*. Keeping those apart matters: raw idf weights average ~4.9
        # against tf-idf's ~0.10, so concatenating them unscaled hands the rq block 98.9%
        # of the row and silently turns any "augment" arm into a "replace" arm.
        return ([(i, v * lex_scale) for i, v in unit(lexical)]
                + [(i, v * rq_scale) for i, v in unit(rq)])

    def build(base, codes, keep, n, prefix_id):
        code_of = {int(i): codes[r] for r, i in enumerate(keep)}
        return [combine(base[i], code_of.get(i), prefix_id) for i in range(n)]

    write_smat(f"{dst}/trn_X_Xf.txt", build(trn_rows, trn_codes, trn_keep, len(trn_rows), xf_id), len(new_Xf))
    write_smat(f"{dst}/tst_X_Xf.txt", build(tst_rows, tst_codes, tst_keep, len(tst_rows), xf_id), len(new_Xf))
    write_smat(f"{dst}/Y_Yf.txt", build(y_rows, lab_codes, lab_keep, len(y_rows), yf_id), len(new_Yf))
    for name, vocab in (("Xf", new_Xf), ("Yf", new_Yf)):
        with open(f"{dst}/{name}.txt", "w") as f:
            f.write("\n".join(vocab) + "\n")
    for fname in LINKED:
        s, d = f"{src}/{fname}", f"{dst}/{fname}"
        if os.path.exists(s) and not os.path.exists(d):
            os.link(s, d)
    print(f"wrote {dst}  ({args.mode}, L={args.levels} K={args.codebook})")


if __name__ == "__main__":
    main()
