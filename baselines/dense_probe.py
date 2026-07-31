"""dense_probe: retrieval-ceiling measurement, dense word-vector retriever vs the
ZestXML lexical shortlist, on the SAME test points / labels.

Point vector  = sum_f tfidf(x,f) * w(f) * mean(vec(tokens of Xf name f)), L2 normalised
Label vector  = sum_g Y_Yf(y,g) * w(g) * mean(vec(tokens of Yf name g)), L2 normalised
               (Yf "1_<tok>" -> <tok>; per-label "__label__i__<name>" -> tokens of <name>)
w(f) is 1 for the plain variant and the SIF weight a/(a+p(token)) for the sif variant.

Outputs recall@K for dense, for the lexical shortlist, and for their union, split into
all / seen / unseen labels; and writes a pure-cosine score matrix for the evaluator.
"""
from __future__ import annotations

import os
import re
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_num_threads(2)

from zestxml.csr import CSR  # noqa: E402
from zestxml.embed import load_word_vectors  # noqa: E402
from zestxml.io import read_bin_smat, read_desc_file, read_text_smat, write_bin_smat  # noqa: E402

SPLIT = re.compile(r"[^a-z0-9]+")


def toks(name: str):
    return [t for t in SPLIT.split(name.lower()) if t]


def label_text(name: str) -> str:
    if name.startswith("1_"):
        return name[2:]
    m = re.match(r"^__label__\d+__(.*)$", name)
    return m.group(1) if m else name


def embed_names(names, index, table, dim):
    """(len(names) x dim) mean of known token vectors; zero if no token known."""
    out = torch.zeros(len(names), dim)
    known = torch.zeros(len(names), dtype=torch.bool)
    for i, n in enumerate(names):
        ids = [index[t] for t in toks(n) if t in index]
        if ids:
            out[i] = table[torch.tensor(ids, dtype=torch.long)].mean(0)
            known[i] = True
    return out, known


def rows_to_vecs(mat: CSR, feat_vecs, feat_w):
    """sum over row non-zeros of value * feat_w * feat_vec, L2 normalised."""
    out = torch.zeros(mat.nrows, feat_vecs.shape[1])
    rid = mat.row_ids()
    w = (mat.values * feat_w[mat.indices]).unsqueeze(1)
    out.index_add_(0, rid, feat_vecs[mat.indices] * w)
    return torch.nn.functional.normalize(out, dim=1)


def rows_to_vecs_onehot(mat: CSR, names, index, V, feat_w):
    """Same as rows_to_vecs but for the identity 'vector' space, without materialising
    an (nnz x V) dense block."""
    ftok = [[index[t] for t in toks(n) if t in index] for n in names]
    cnt = torch.tensor([len(x) for x in ftok], dtype=torch.long)
    flat = torch.tensor([t for x in ftok for t in x], dtype=torch.long)
    start = torch.zeros(len(ftok) + 1, dtype=torch.long)
    start[1:] = cnt.cumsum(0)

    out = torch.zeros(mat.nrows * V)
    rid = mat.row_ids()
    w = mat.values * feat_w[mat.indices]
    fc = cnt[mat.indices]
    keep = fc > 0
    rid, w, fidx, fc = rid[keep], w[keep], mat.indices[keep], fc[keep]
    seg = torch.repeat_interleave(fc)
    offs = torch.zeros(fc.numel() + 1, dtype=torch.long)
    offs[1:] = fc.cumsum(0)
    pos = torch.arange(int(fc.sum())) - offs[seg] + start[fidx][seg]
    out.index_add_(0, rid[seg] * V + flat[pos], (w / fc.float())[seg])
    return torch.nn.functional.normalize(out.view(mat.nrows, V), dim=1)


def topk_from_dense(Q, L, k, chunk=1024):
    idxs, vals = [], []
    for lo in range(0, Q.shape[0], chunk):
        s = Q[lo : lo + chunk] @ L.t()
        v, i = torch.topk(s, k, dim=1)
        idxs.append(i)
        vals.append(v)
    return torch.cat(idxs), torch.cat(vals)


def topk_from_csr(mat: CSR, k, n_labels):
    """Top-k columns per row of a sparse score matrix, padded with -1."""
    idx = torch.full((mat.nrows, k), -1, dtype=torch.long)
    ip = mat.indptr
    for r in range(mat.nrows):
        lo, hi = int(ip[r]), int(ip[r + 1])
        if hi == lo:
            continue
        v = mat.values[lo:hi]
        kk = min(k, hi - lo)
        top = torch.topk(v, kk).indices
        idx[r, :kk] = mat.indices[lo:hi][top]
    return idx


def recall(cand_idx, truth: CSR, mask):
    """mean over points with >=1 masked true label of |cand & true| / |true|."""
    n, k = cand_idx.shape
    hit = torch.zeros(n)
    tot = torch.zeros(n)
    ip = truth.indptr
    for r in range(n):
        lo, hi = int(ip[r]), int(ip[r + 1])
        ys = truth.indices[lo:hi]
        ys = ys[mask[ys]]
        if ys.numel() == 0:
            continue
        tot[r] = ys.numel()
        c = cand_idx[r]
        c = c[c >= 0]
        hit[r] = torch.isin(ys, c).sum()
    ok = tot > 0
    macro = 100.0 * (hit[ok] / tot[ok]).mean().item()
    micro = 100.0 * hit.sum().item() / max(tot.sum().item(), 1e-9)
    return macro, micro, int(ok.sum())


def main(data_dir, key, vec_path, sif_a=0.0, K=100):
    t0 = time.time()
    Xf = read_desc_file(f"{data_dir}/Xf.txt")
    Yf = read_desc_file(f"{data_dir}/Yf.txt")
    tst_X_Xf = read_text_smat(f"{data_dir}/tst_X_Xf.txt")
    trn_X_Xf = read_text_smat(f"{data_dir}/trn_X_Xf.txt")
    Y_Yf = read_text_smat(f"{data_dir}/Y_Yf.txt")
    tst_X_Y = read_text_smat(f"{data_dir}/tst_X_Y.txt")
    trn_X_Y = read_text_smat(f"{data_dir}/trn_X_Y.txt")
    n_lab = Y_Yf.nrows
    print(f"points {tst_X_Xf.nrows} labels {n_lab} Xf {len(Xf)} Yf {len(Yf)}", flush=True)

    yf_text = [label_text(n) for n in Yf]
    vocab = {t for n in Xf for t in toks(n)} | {t for n in yf_text for t in toks(n)}
    if vec_path == "onehot":
        # control: identity "vectors" -> plain lexical cosine in the shared token space
        index = {t: i for i, t in enumerate(sorted(vocab))}
        table = torch.zeros(1, len(index))  # placeholder: only its width is used
    elif vec_path.startswith("numberbatch:"):
        p = vec_path.split(":", 1)[1]
        pref = {"/c/en/" + t for t in vocab}
        idx0, table = load_word_vectors(p, vocab=pref)
        index = {k[len("/c/en/"):]: v for k, v in idx0.items()}
    else:
        index, table = load_word_vectors(vec_path, vocab=vocab)
    dim = table.shape[1]
    print(f"vectors: {len(index)}/{len(vocab)} tokens covered "
          f"({len(index)/len(vocab):.1%}) dim={dim}  [{time.time()-t0:.0f}s]", flush=True)
    table = torch.nn.functional.normalize(table, dim=1)

    if vec_path == "onehot":
        xf_vec = yf_vec = None
        xf_known = torch.tensor([any(t in index for t in toks(n)) for n in Xf])
        yf_known = torch.tensor([any(t in index for t in toks(n)) for n in yf_text])
    else:
        xf_vec, xf_known = embed_names(Xf, index, table, dim)
        yf_vec, yf_known = embed_names(yf_text, index, table, dim)
    print(f"name coverage: Xf {xf_known.float().mean():.1%}  Yf {yf_known.float().mean():.1%}",
          flush=True)

    # SIF-style down-weighting by corpus document frequency of the feature
    if sif_a > 0:
        df_x = torch.zeros(len(Xf)).index_add_(0, trn_X_Xf.indices, torch.ones(trn_X_Xf.nnz))
        p_x = df_x / max(1.0, float(df_x.sum()))
        w_x = sif_a / (sif_a + p_x)
        df_y = torch.zeros(len(Yf)).index_add_(0, Y_Yf.indices, torch.ones(Y_Yf.nnz))
        p_y = df_y / max(1.0, float(df_y.sum()))
        w_y = sif_a / (sif_a + p_y)
    else:
        w_x = torch.ones(len(Xf))
        w_y = torch.ones(len(Yf))

    if vec_path == "onehot":
        Q = rows_to_vecs_onehot(tst_X_Xf, Xf, index, dim, w_x)
        L = rows_to_vecs_onehot(Y_Yf, yf_text, index, dim, w_y)
    else:
        Q = rows_to_vecs(tst_X_Xf, xf_vec, w_x)
        L = rows_to_vecs(Y_Yf, yf_vec, w_y)
    print(f"embedded. zero query rows {(Q.norm(dim=1)==0).sum()}  "
          f"zero label rows {(L.norm(dim=1)==0).sum()}  [{time.time()-t0:.0f}s]", flush=True)

    unseen = torch.zeros(n_lab, dtype=torch.bool)
    with open(f"{data_dir}/unseen_labels.txt") as f:
        for line in f:
            if line.strip():
                unseen[int(line.split()[0])] = True
    allm = torch.ones(n_lab, dtype=torch.bool)

    d_idx, d_val = topk_from_dense(Q, L, K)
    lex_path = None
    for p in (f"Results/Npm2exact/shortlist.bin", f"Results/{key}_lex/shortlist.bin"):
        if os.path.exists(p):
            lex_path = p
            break
    lex_idx = None
    if lex_path and data_dir.endswith("GZ-NPM"):
        lex = read_bin_smat(lex_path)
        assert lex.shape == tst_X_Y.shape, f"{lex.shape} vs {tst_X_Y.shape}"
        lex_idx = topk_from_csr(lex, K, n_lab)
        lex_idx50 = topk_from_csr(lex, 50, n_lab)
    print(f"[{time.time()-t0:.0f}s] candidates built", flush=True)

    print("\n== recall@%d (mean over points with >=1 true label of that kind) ==" % K)
    print("%-26s %9s %9s %9s" % ("method", "all", "seen", "unseen"))

    def line(name, idx):
        a, am, na = recall(idx, tst_X_Y, allm)
        s, sm, ns = recall(idx, tst_X_Y, ~unseen)
        u, um, nu = recall(idx, tst_X_Y, unseen)
        print("%-26s %8.2f %8.2f %8.2f  | micro %6.2f %6.2f %6.2f" %
              (name, a, s, u, am, sm, um), flush=True)
        return dict(macro=(a, s, u), micro=(am, sm, um))

    res = {}
    res["dense@%d" % K] = line("dense top-%d" % K, d_idx)
    if lex_idx is not None:
        res["lex@%d" % K] = line("lexical shortlist top-%d" % K, lex_idx)
        union = torch.cat([d_idx[:, :50], lex_idx50], dim=1)
        res["union50"] = line("union dense50+lex50", union)
        res["dense@50"] = line("dense top-50", d_idx[:, :50])
        res["lex@50"] = line("lexical top-50", lex_idx50)

    # secondary: pure dense cosine as a ranker
    out_dir = f"Results/{key}"
    os.makedirs(out_dir, exist_ok=True)
    rows = torch.arange(d_idx.shape[0]).unsqueeze(1).expand_as(d_idx).reshape(-1)
    smat = CSR.from_coo(rows, d_idx.reshape(-1), d_val.reshape(-1).clamp(min=1e-6),
                        (d_idx.shape[0], n_lab))
    write_bin_smat(smat, f"{out_dir}/score_mat.bin")
    print(f"wrote {out_dir}/score_mat.bin  [{time.time()-t0:.0f}s]", flush=True)
    return res


if __name__ == "__main__":
    data_dir = sys.argv[1]
    key = sys.argv[2]
    vec = sys.argv[3]
    sif = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
    main(data_dir, key, vec, sif)
