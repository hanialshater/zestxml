"""Standalone evaluation with seen/unseen label splits (no xclib dependency).

Usage: python semantic_atoms/evaluate.py <dataset-name> [<dataset-name> ...]
Reads Results/<name>/score_mat.bin and GZXML-Datasets/<name>/{tst_X_Y.txt,
trn_X_Y.txt,unseen_labels.txt}; prints P@k overall and restricted to
seen-only / unseen-only label subsets, plus unseen recall@k.
"""
import sys, os
import numpy as np
import scipy.sparse as sp

def read_sparse_mat(filename):
    with open(filename) as f:
        nr, nc = map(int, f.readline().split())
        data, indices, indptr = [], [], [0]
        for line in f:
            for tok in line.split():
                j, v = tok.split(':')
                indices.append(int(j)); data.append(float(v))
            indptr.append(len(data))
    return sp.csr_matrix((data, indices, indptr), shape=(nr, nc), dtype=np.float32)

def readbuf(buf, dtype, offset=0, count=1):
    val = np.frombuffer(buf, offset=offset, dtype=dtype, count=count)
    if count == 1: val = val[0]
    offset += np.dtype(dtype).itemsize * count
    return val, offset

def read_bin_spmat(fname):
    buf = open(fname, 'rb').read()
    (nr, nc), offset = readbuf(buf, np.int32, 0, 2)
    nsz, offset = readbuf(buf, np.int64, offset, 1)
    size, offset = readbuf(buf, np.int32, offset, nsz)
    indptr = np.zeros(nr + 1, np.int64); indptr[1:] = np.asarray(size).cumsum()
    n = int(indptr[-1])
    # (index, value) pairs are interleaved int32/float32: read buffer twice
    temp, _ = readbuf(buf, np.int32, offset, 2 * n)
    inds = temp.reshape(-1, 2)[:, 0]
    temp, offset = readbuf(buf, np.float32, offset, 2 * n)
    data = temp.reshape(-1, 2)[:, 1]
    return sp.csr_matrix((data, inds, indptr), (nr, nc))

def precision_at_k(score, gt, ks=(1, 3, 5)):
    """Mean P@k over rows of `gt` that have at least one positive."""
    rows = np.where(np.asarray(gt.sum(1)).ravel() > 0)[0]
    out = {}
    kmax = max(ks)
    hits = np.zeros((len(rows), kmax))
    for r, i in enumerate(rows):
        s = score.getrow(i)
        if s.nnz == 0: continue
        order = s.indices[np.argsort(-s.data)][:kmax]
        g = set(gt.getrow(i).indices)
        hits[r, :len(order)] = [j in g for j in order]
    for k in ks:
        out[f'P@{k}'] = 100 * hits[:, :k].mean(1).mean()
    return out, len(rows)

def recall_at_k(score, gt, ks=(5, 10)):
    rows = np.where(np.asarray(gt.sum(1)).ravel() > 0)[0]
    out = {}
    for k in ks:
        rec = []
        for i in rows:
            s = score.getrow(i)
            order = s.indices[np.argsort(-s.data)][:k]
            g = set(gt.getrow(i).indices)
            rec.append(len(g & set(order)) / len(g))
        out[f'R@{k}'] = 100 * float(np.mean(rec)) if rec else 0.0
    return out

def col_restrict(mat, cols_keep):
    m = sp.diags(cols_keep.astype(np.float32))
    r = (mat @ m).tocsr(); r.eliminate_zeros()
    return r

def evaluate(name):
    ddir, rdir = f'GZXML-Datasets/{name}', f'Results/{name}'
    tst_X_Y = read_sparse_mat(f'{ddir}/tst_X_Y.txt')
    score = read_bin_spmat(f'{rdir}/score_mat.bin')
    unseen = np.zeros(tst_X_Y.shape[1], bool)
    fn = f'{ddir}/unseen_labels.txt'
    if os.path.exists(fn):
        unseen[[int(l) for l in open(fn)]] = True
    seen = ~unseen

    print(f'\n=== {name} ===')
    p, n = precision_at_k(score, tst_X_Y)
    print(f'overall  ({n} docs): ' + '  '.join(f'{k}={v:.2f}' for k, v in p.items()))
    p, n = precision_at_k(col_restrict(score, seen), col_restrict(tst_X_Y, seen))
    print(f'seen     ({n} docs): ' + '  '.join(f'{k}={v:.2f}' for k, v in p.items()))
    su, gu = col_restrict(score, unseen), col_restrict(tst_X_Y, unseen)
    p, n = precision_at_k(su, gu)
    r = recall_at_k(su, gu)
    print(f'unseen   ({n} docs): ' + '  '.join(f'{k}={v:.2f}' for k, v in p.items())
          + '  ' + '  '.join(f'{k}={v:.2f}' for k, v in r.items()))

if __name__ == '__main__':
    for name in sys.argv[1:]:
        evaluate(name)
