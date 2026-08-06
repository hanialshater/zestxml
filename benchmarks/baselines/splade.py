"""SPLADE-style learned sparse retrieval for GZXML, without any pretrained LM.

Architecture (the SPLADE *mechanism*, learned from scratch):
  point side  : e_p = relu( x + expand_x(x) ),  expand_x = x @ A @ B   (low rank)
  label side  : e_l = relu( y @ M0 + expand_y(y) ), expand_y = y @ C @ D
                M0 is the fixed name-match map label-feature -> point-feature
  score(p, l) = <topk(e_p), e_l>   (inverted-index dot product)

Trained on sampled (point, label) pairs from trn_X_Y with BCE + FLOPS(L1) regulariser.
Inference sparsifies both sides to top-k and retrieves with scipy sparse (inverted index).
"""
from __future__ import annotations

import argparse, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from zestxml.csr import CSR
from zestxml.io import read_text_smat, read_desc_file, write_bin_smat, ensure_dir

torch.set_num_threads(2)


def to_scipy(m: CSR) -> sp.csr_matrix:
    return sp.csr_matrix(
        (m.values.numpy(), m.indices.numpy(), m.indptr.numpy()), shape=m.shape)


def build_M0(xf, yf):
    """Fixed label-feature -> point-feature map from feature names ("1_<tok>" -> "<tok>")."""
    xidx = {n: i for i, n in enumerate(xf)}
    rows, cols = [], []
    for j, n in enumerate(yf):
        if n.startswith("1_"):
            t = n[2:]
            if t in xidx:
                rows.append(j); cols.append(xidx[t])
    return rows, cols


class Splade(nn.Module):
    def __init__(self, nXf, nYf, Ve, rank):
        super().__init__()
        self.nXf, self.Ve = nXf, Ve
        self.A = nn.Parameter(torch.randn(nXf, rank) * 0.05)
        self.B = nn.Parameter(torch.randn(rank, Ve) * 0.05)
        self.C = nn.Parameter(torch.randn(nYf, rank) * 0.05)
        self.D = nn.Parameter(torch.randn(rank, Ve) * 0.05)
        self.scale = nn.Parameter(torch.tensor(4.0))
        self.bias = nn.Parameter(torch.tensor(-3.0))

    def _exp(self, sp_batch, base_dense, A, B):
        h = torch.sparse.mm(sp_batch, A)          # (B, rank)
        e = base_dense.clone()
        e[:, : self.Ve] = e[:, : self.Ve] + h @ B
        return torch.relu(e)

    def points(self, sp_batch, base_dense):
        return self._exp(sp_batch, base_dense, self.A, self.B)

    def labels(self, sp_batch, base_dense):
        return self._exp(sp_batch, base_dense, self.C, self.D)


def sparse_rows(mat: sp.csr_matrix, rows, ncol):
    """torch sparse_coo of mat[rows] (for the low-rank projection)."""
    sub = mat[rows]
    coo = sub.tocoo()
    idx = torch.from_numpy(np.vstack([coo.row, coo.col])).long()
    return torch.sparse_coo_tensor(idx, torch.from_numpy(coo.data).float(),
                                   (sub.shape[0], ncol)).coalesce()


def dense_rows(mat: sp.csr_matrix, rows, ncol):
    sub = mat[rows].toarray()
    return torch.from_numpy(sub).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-data", default="GZXML-Datasets/GZ-NPM")
    ap.add_argument("-res", default="Results/splade")
    ap.add_argument("-rank", type=int, default=128)
    ap.add_argument("-Ve", type=int, default=4096)
    ap.add_argument("-epochs", type=int, default=3)
    ap.add_argument("-bs", type=int, default=256)
    ap.add_argument("-ncand", type=int, default=768)
    ap.add_argument("-ktrain", type=int, default=96)
    ap.add_argument("-kinfer", type=int, default=200)
    ap.add_argument("-l1p", type=float, default=3e-4)
    ap.add_argument("-l1l", type=float, default=3e-4)
    ap.add_argument("-lr", type=float, default=2e-3)
    ap.add_argument("-posw", type=float, default=20.0)
    ap.add_argument("-budget", type=float, default=1e9, help="seconds of training")
    # the two fixes the first run's diagnosis called for
    ap.add_argument("-drop_label_id", type=int, default=0,
                    help="zero out the per-label __label__i__ features so the label side "
                         "must route through shared tokens instead of memorising an id")
    ap.add_argument("-norm_labels", type=int, default=0,
                    help="L2-normalise label vectors at inference, killing the norm/"
                         "popularity prior that lets memorised seen labels outrank unseen")
    a = ap.parse_args()
    t0 = time.time()

    D = a.data
    trnX = read_text_smat(f"{D}/trn_X_Xf.txt").unit_normalize_rows()
    tstX = read_text_smat(f"{D}/tst_X_Xf.txt").unit_normalize_rows()
    Y = read_text_smat(f"{D}/Y_Yf.txt").unit_normalize_rows()
    trnXY = read_text_smat(f"{D}/trn_X_Y.txt")
    xf, yf = read_desc_file(f"{D}/Xf.txt"), read_desc_file(f"{D}/Yf.txt")
    if a.drop_label_id:
        # a per-label feature is a free parameter for every label that has positives, and
        # gradient descent memorises into it rather than learning the shared token map;
        # the first run traded unseen P@1 36.79 -> 23.00 doing exactly that
        keep = torch.tensor([not n.startswith("__label__") for n in yf])
        Y = Y.subset(keep[Y.indices])
        print(f"dropped {int((~keep).sum())} per-label id features", flush=True)
    nP, nXf = trnX.shape
    nL, nYf = Y.shape
    print(f"train {trnX.shape} test {tstX.shape} labels {Y.shape}", flush=True)

    # ---- reorder point features so the Ve most frequent come first (expansion target)
    dfreq = np.bincount(trnX.indices.numpy(), minlength=nXf)
    order = np.argsort(-dfreq)
    perm = np.empty(nXf, dtype=np.int64)
    perm[order] = np.arange(nXf)          # old id -> new id
    Ve = min(a.Ve, nXf)

    def remap(m: CSR):
        s = to_scipy(m)
        s.indices = perm[s.indices].astype(np.int32)
        s.sort_indices()
        return s

    Xtr, Xte = remap(trnX), remap(tstX)
    Ysp = to_scipy(Y)

    # ---- fixed name-match base map for labels
    r, c = build_M0(xf, yf)
    print(f"M0 name-match entries: {len(r)} / {nYf} label features", flush=True)
    M0 = sp.csr_matrix((np.ones(len(r), np.float32), (r, perm[np.array(c)])),
                       shape=(nYf, nXf))
    Lbase = (Ysp @ M0).tocsr()            # (nL, nXf) label base vector in point-feat space

    XY = to_scipy(trnXY)
    pos_lists = [XY.indices[XY.indptr[i]:XY.indptr[i + 1]] for i in range(nP)]
    has_pos = np.array([len(p) > 0 for p in pos_lists])
    train_pts = np.nonzero(has_pos)[0]
    print(f"{len(train_pts)} points with >=1 positive", flush=True)

    model = Splade(nXf, nYf, Ve, a.rank)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    rng = np.random.default_rng(0)

    step = 0
    stop = False
    for ep in range(a.epochs):
        pts = rng.permutation(train_pts)
        for s in range(0, len(pts), a.bs):
            b = pts[s:s + a.bs]
            if len(b) < 8:
                continue
            posu = np.unique(np.concatenate([pos_lists[i] for i in b]))
            if len(posu) > a.ncand:
                posu = rng.choice(posu, a.ncand, replace=False)
            neg = rng.integers(0, nL, size=max(0, a.ncand - len(posu)))
            cand = np.unique(np.concatenate([posu, neg]))
            posset = {l: j for j, l in enumerate(cand)}

            tgt = torch.zeros(len(b), len(cand))
            for bi, i in enumerate(b):
                for l in pos_lists[i]:
                    j = posset.get(l)
                    if j is not None:
                        tgt[bi, j] = 1.0
            if tgt.sum() == 0:
                continue

            ep_ = model.points(sparse_rows(Xtr, b, nXf), dense_rows(Xtr, b, nXf))
            el_ = model.labels(sparse_rows(Ysp, cand, nYf), dense_rows(Lbase, cand, nXf))

            k = min(a.ktrain, nXf)
            vals, idx = torch.topk(ep_, k, dim=1)
            g = el_.t()[idx.reshape(-1)].reshape(len(b), k, len(cand))
            sc = torch.bmm(vals.unsqueeze(1), g).squeeze(1)   # (B, C)

            logit = model.scale * sc + model.bias
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logit, tgt, pos_weight=torch.tensor(a.posw))
            # FLOPS regulariser (SPLADE): squared mean activation per dimension
            loss = loss + a.l1p * (ep_.mean(0) ** 2).sum() + a.l1l * (el_.mean(0) ** 2).sum()

            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
            if step % 20 == 0:
                nzp = (ep_ > 0).float().sum(1).mean().item()
                nzl = (el_ > 0).float().sum(1).mean().item()
                print(f"ep{ep} step{step} loss {loss.item():.4f} nnz_p {nzp:.0f} "
                      f"nnz_l {nzl:.0f} t {time.time()-t0:.0f}s", flush=True)
            if time.time() - t0 > a.budget:
                print("TIME BUDGET HIT -- stopping training", flush=True)
                stop = True
                break
        if stop:
            break

    # ------------------------------------------------------------------ inference
    model.eval()
    stats = {}

    def encode(Xmat, base, sparsemat, nfeat_in, is_label, k):
        rows_out, cols_out, vals_out = [], [], []
        n = Xmat.shape[0]
        dense_nnz = 0.0
        for s in range(0, n, 512):
            r_ = np.arange(s, min(s + 512, n))
            with torch.no_grad():
                fn = model.labels if is_label else model.points
                e = fn(sparse_rows(sparsemat, r_, nfeat_in), dense_rows(base, r_, nXf))
                dense_nnz += (e > 0).float().sum().item()
                kk = min(k, e.shape[1])
                v, i = torch.topk(e, kk, dim=1)
                m = v > 0
                rr = torch.arange(len(r_)).unsqueeze(1).expand_as(i)[m] + s
                rows_out.append(rr.numpy()); cols_out.append(i[m].numpy())
                vals_out.append(v[m].numpy())
        out = sp.csr_matrix((np.concatenate(vals_out),
                             (np.concatenate(rows_out), np.concatenate(cols_out))),
                            shape=(n, nXf))
        return out, dense_nnz / n

    tE, nzp = encode(Xte, Xte, Xte, nXf, False, a.kinfer)
    lE, nzl = encode(Ysp, Lbase, Ysp, nYf, True, a.kinfer)
    if a.norm_labels:
        norms = np.sqrt(lE.multiply(lE).sum(1)).A.ravel()
        norms[norms == 0] = 1.0
        lE = sp.diags(1.0 / norms) @ lE
        lE = lE.tocsr()
        print("L2-normalised label vectors", flush=True)
    stats["dense_nnz_point"] = nzp
    stats["dense_nnz_label"] = nzl
    stats["topk_nnz_point"] = tE.nnz / tE.shape[0]
    stats["topk_nnz_label"] = lE.nnz / lE.shape[0]
    print("SPARSITY", stats, flush=True)

    S = (tE @ lE.T).tocsr()               # inverted-index retrieval
    print(f"score nnz {S.nnz} ({S.nnz/S.shape[0]:.0f}/point) t {time.time()-t0:.0f}s", flush=True)
    # keep top 100 per row to keep the file small
    keepr, keepc, keepv = [], [], []
    for i in range(S.shape[0]):
        lo, hi = S.indptr[i], S.indptr[i + 1]
        d, ind = S.data[lo:hi], S.indices[lo:hi]
        if len(d) > 100:
            sel = np.argpartition(-d, 100)[:100]
            d, ind = d[sel], ind[sel]
        keepr.append(np.full(len(d), i)); keepc.append(ind); keepv.append(d)
    rr = np.concatenate(keepr); cc = np.concatenate(keepc); vv = np.concatenate(keepv)
    out = CSR.from_coo(torch.from_numpy(rr).long(), torch.from_numpy(cc).long(),
                       torch.from_numpy(vv).float(), (tE.shape[0], nL))
    ensure_dir(a.res)
    write_bin_smat(out, f"{a.res}/score_mat.bin")
    print(f"wrote {a.res}/score_mat.bin  total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
