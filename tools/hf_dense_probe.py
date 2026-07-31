"""Does a pretrained encoder raise the candidate-generation ceiling?

    python tools/hf_dense_probe.py --data GZXML-Datasets/GZ-NPM \
        --lexical Results/npm/shortlist.bin --out Results/npm-hybrid

This is the measurement the CPU sweep could not make: HuggingFace is blocked from the
sandbox this was written in, so the best available encoder was a static word-vector table,
and it recalled *less* than lexical matching (39.3% vs 71.8% at K=100). The whole question
is whether a real sentence encoder changes that.

What it reports, all at an equal candidate budget so the comparison is fair:

    lexical top-K        the ZestXML shortlist, the number to beat
    dense top-K          the encoder alone
    hybrid top-K/2 each  the union

split by seen and unseen labels, because that split is the point. It then writes the
hybrid shortlist, which `run_torch.py -shortlist_file <path>` consumes, so the downstream
effect on P@1/PSP can be measured rather than guessed.

Needs: pip install sentence-transformers
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zestxml.csr import CSR, counts_to_indptr  # noqa: E402
from zestxml.io import ensure_dir, read_bin_smat, read_text_smat, write_bin_smat  # noqa: E402


def read_lines(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        return [line.rstrip("\n") for line in f]


def encode(texts, model_name, batch_size, device, label=""):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device=device)
    t0 = time.time()
    vecs = model.encode(
        texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True,
        show_progress_bar=True,
    )
    print(f"  encoded {len(texts)} {label} texts in {time.time() - t0:.0f}s -> {vecs.shape}")
    return torch.from_numpy(vecs)


def topk_dense(points, labels, k, chunk=256):
    """Top-k labels per point by cosine, both sides already L2 normalised."""
    idx_out, val_out = [], []
    for lo in range(0, points.shape[0], chunk):
        sims = points[lo : lo + chunk] @ labels.t()
        v, i = torch.topk(sims, min(k, labels.shape[0]), dim=1)
        idx_out.append(i)
        val_out.append(v)
    return torch.cat(idx_out), torch.cat(val_out)


def recall(candidates: CSR, truth: CSR, mask=None):
    """Micro recall: what fraction of true (point, label) pairs are in the candidate set.

    ``mask`` selects a subset of labels (used for the seen / unseen split).
    """
    t_rows, t_cols = truth.row_ids(), truth.indices
    if mask is not None:
        keep = mask[t_cols]
        t_rows, t_cols = t_rows[keep], t_cols[keep]
    if t_rows.numel() == 0:
        return float("nan"), 0
    truth_keys = t_rows * truth.ncols + t_cols
    hit = torch.isin(truth_keys, candidates.keys())
    return 100.0 * float(hit.sum()) / t_rows.numel(), int(t_rows.numel())


def as_csr(idx, val, n_labels):
    n = idx.shape[0]
    counts = torch.full((n,), idx.shape[1], dtype=torch.long)
    order = torch.argsort(idx, dim=1)  # keep each row's candidates ordered by label id
    return CSR(
        counts_to_indptr(counts),
        torch.gather(idx, 1, order).reshape(-1),
        torch.gather(val, 1, order).reshape(-1),
        (n, n_labels),
    )


def union_shortlists(a: CSR, b: CSR):
    """Union of two candidate sets, keeping each side's own score."""
    keys = torch.cat([a.keys(), b.keys()])
    vals = torch.cat([a.values, b.values])
    order = torch.argsort(keys)
    keys, vals = keys[order], vals[order]
    uniq, inv = torch.unique(keys, return_inverse=True)
    best = torch.zeros(uniq.numel(), dtype=vals.dtype).scatter_reduce(
        0, inv, vals, reduce="amax", include_self=False
    )
    return CSR.from_sorted_keys(uniq, best, a.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="GZXML-Datasets/GZ-NPM")
    ap.add_argument("--lexical", default="", help="shortlist.bin from a ZestXML predict run")
    ap.add_argument("--out", default="Results/hybrid")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--k", type=int, default=100, help="total candidate budget")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    truth = read_text_smat(f"{a.data}/tst_X_Y.txt")
    n_labels = truth.ncols
    unseen = torch.zeros(n_labels, dtype=torch.bool)
    path = f"{a.data}/unseen_labels.txt"
    if os.path.exists(path):
        for line in read_lines(path):
            if line.strip():
                unseen[int(line.split()[0])] = True
    print(f"{a.data}: {truth.nrows} test points, {n_labels} labels, {int(unseen.sum())} unseen")

    # label text: Y.txt if the dataset carries it, else the label-feature names
    label_texts = read_lines(f"{a.data}/Y.txt")
    test_texts = read_lines(f"{a.data}/tst_X.txt")
    assert len(label_texts) == n_labels, f"Y.txt has {len(label_texts)} rows, need {n_labels}"
    assert len(test_texts) == truth.nrows, f"tst_X.txt has {len(test_texts)} rows"

    print(f"encoding with {a.model} on {a.device}")
    lab_vecs = encode(label_texts, a.model, a.batch_size, a.device, "label")
    pt_vecs = encode(test_texts, a.model, a.batch_size, a.device, "test point")

    rows = []

    dense_idx, dense_val = topk_dense(pt_vecs, lab_vecs, a.k)
    dense = as_csr(dense_idx, dense_val, n_labels)
    rows.append(("dense top-%d" % a.k, dense))

    lexical = None
    if a.lexical:
        lexical = read_bin_smat(a.lexical)
        assert lexical.shape == truth.shape, f"{lexical.shape} != {truth.shape}"
        rows.append(("lexical top-%d" % a.k, lexical))

        half_idx, half_val = topk_dense(pt_vecs, lab_vecs, a.k // 2)
        dense_half = as_csr(half_idx, half_val, n_labels)
        # keep the lexical side's own top half, by score
        lex_half = lexical  # already ~K per row; trimming needs its scores, keep as is
        hybrid = union_shortlists(lex_half, dense_half)
        rows.append(("hybrid (lexical + dense top-%d)" % (a.k // 2), hybrid))

    print(f"\n{'candidate set':<38} {'all':>8} {'seen':>8} {'unseen':>8} {'cands/pt':>9}")
    for name, cand in rows:
        all_r, _ = recall(cand, truth)
        seen_r, _ = recall(cand, truth, ~unseen)
        uns_r, _ = recall(cand, truth, unseen)
        print(f"{name:<38} {all_r:8.2f} {seen_r:8.2f} {uns_r:8.2f} {cand.nnz / cand.nrows:9.1f}")

    if lexical is not None:
        ensure_dir(a.out)
        out_path = f"{a.out}/hybrid_shortlist.bin"
        write_bin_smat(rows[-1][1], out_path)
        print(f"\nwrote {out_path}")
        print("feed it to the pipeline with:  -shortlist_file %s" % out_path)


if __name__ == "__main__":
    main()
