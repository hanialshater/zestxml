"""label_graph: propagate XMC scores over the label co-occurrence graph.

S' = (1-a) S + a S G   (optionally iterated), G = row-normalised top-k cosine
co-occurrence graph over label columns of trn_X_Y.
"""
import os, sys, argparse
import numpy as np, scipy.sparse as sp, torch
sys.path.insert(0, "/home/user/zestxml")
torch.set_num_threads(2)
from zestxml.io import read_text_smat, read_bin_smat, write_bin_smat, ensure_dir
from zestxml.csr import CSR


def csr_to_scipy(m):
    return sp.csr_matrix((m.values.numpy().astype(np.float64), m.indices.numpy(), m.indptr.numpy()),
                         shape=m.shape)


def scipy_to_csr(m):
    m = m.tocsr(); m.sort_indices()
    return CSR(indptr=torch.as_tensor(m.indptr.astype(np.int64)),
               indices=torch.as_tensor(m.indices.astype(np.int64)),
               values=torch.as_tensor(m.data.astype(np.float32)),
               shape=m.shape)


def build_graph(Y, k=15, mode="cosine", pmi_shift=0.0, selfloop_isolated=False):
    """Y: (n_points x n_labels) scipy csr binary. Returns row-normalised (L x L) csr."""
    Yb = (Y > 0).astype(np.float64)
    C = (Yb.T @ Yb).tocsr()           # co-occurrence counts
    deg = np.asarray(Yb.sum(0)).ravel()
    C.setdiag(0); C.eliminate_zeros()
    C = C.tocoo()
    r, c, v = C.row, C.col, C.data
    if mode == "cosine":
        w = v / np.sqrt(np.maximum(deg[r] * deg[c], 1e-9))
    else:  # ppmi
        N = Yb.shape[0]
        w = np.log(np.maximum(v * N / np.maximum(deg[r] * deg[c], 1e-9), 1e-12)) - pmi_shift
        w = np.maximum(w, 0.0)
    G = sp.csr_matrix((w, (r, c)), shape=C.shape)
    G.eliminate_zeros()
    # keep top-k per row
    G = topk_rows(G, k)
    rs = np.asarray(G.sum(1)).ravel()
    rs[rs == 0] = 1.0
    G = sp.diags(1.0 / rs) @ G
    if selfloop_isolated:
        # labels with no incoming edge (e.g. every unseen label) would otherwise be
        # shrunk by (1-a)^steps relative to labels the graph can feed; a self loop
        # makes propagation a no-op for them instead of a penalty.
        indeg = np.asarray((G != 0).sum(0)).ravel()
        iso = np.where(indeg == 0)[0]
        G = (G + sp.csr_matrix((np.ones(len(iso)), (iso, iso)), shape=G.shape)).tocsr()
    return G


def topk_rows(G, k):
    G = G.tocsr()
    keep = []
    ip, idx, dat = G.indptr, G.indices, G.data
    for i in range(G.shape[0]):
        lo, hi = ip[i], ip[i + 1]
        if hi - lo <= k:
            keep.append(np.arange(lo, hi))
        else:
            sel = np.argpartition(-dat[lo:hi], k)[:k]
            keep.append(lo + sel)
    keep = np.concatenate(keep) if keep else np.zeros(0, dtype=int)
    rows = np.repeat(np.arange(G.shape[0]), np.diff(ip))[keep]
    return sp.csr_matrix((dat[keep], (rows, idx[keep])), shape=G.shape)


def propagate(S, G, a, steps=1, restrict=True):
    out = S.copy()
    cur = S
    for _ in range(steps):
        cur = cur @ G
        out = (1 - a) * out + a * cur
    if restrict:
        mask = (S != 0)
        out = out.multiply(mask)
    out = out.tocsr(); out.eliminate_zeros()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-data", default="/home/user/zestxml/GZXML-Datasets/GZ-NPM")
    p.add_argument("-trn_X_Y", default=None)
    p.add_argument("-scores", required=True)
    p.add_argument("-out", required=True)
    p.add_argument("-k", type=int, default=15)
    p.add_argument("-alpha", type=float, default=0.2)
    p.add_argument("-steps", type=int, default=1)
    p.add_argument("-mode", default="cosine")
    p.add_argument("-restrict", type=int, default=1)
    p.add_argument("-selfloop", type=int, default=0)
    p.add_argument("-stats", type=int, default=0)
    args = p.parse_args()

    trn_X_Y = args.trn_X_Y or os.path.join(args.data, "trn_X_Y.txt")
    Y = csr_to_scipy(read_text_smat(trn_X_Y))
    G = build_graph(Y, k=args.k, mode=args.mode, selfloop_isolated=bool(args.selfloop))
    if args.stats:
        unseen = [int(l.split()[0]) for l in open(os.path.join(args.data, "unseen_labels.txt"))]
        nb = np.diff(G.indptr)
        print("labels=%d  labels with >=1 neighbour=%d" % (G.shape[0], (nb > 0).sum()))
        print("unseen labels=%d  unseen with >=1 neighbour=%d" % (len(unseen), (nb[unseen] > 0).sum()))
        indeg = np.asarray((G > 0).sum(0)).ravel()
        print("unseen with >=1 IN-edge (reachable by propagation)=%d" % (indeg[unseen] > 0).sum())
        print("mean out-degree (seen labels)=%.2f" % nb[nb > 0].mean())
    S = csr_to_scipy(read_bin_smat(args.scores))
    Sp = propagate(S, G, args.alpha, args.steps, bool(args.restrict))
    ensure_dir(os.path.dirname(args.out))
    write_bin_smat(scipy_to_csr(Sp), args.out)
    print("wrote", args.out, "nnz", Sp.nnz)


if __name__ == "__main__":
    main()
