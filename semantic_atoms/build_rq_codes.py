"""Ablation: semantic-ID-style codes through the SAME harness as build_atoms.

Residual k-means (RQ-VAE style, without the VAE) on the same LSA embeddings:
L levels x C centroids, each doc/label gets one code per level. Codes become
features '#rq<level>c<id>' on both sides, same weighting scheme as the atoms.
The only difference vs build_atoms.py is the shape of the sparse basis:
a tree path (L active out of L*C, non-compositional) instead of a flat
overcomplete k-sparse code.
"""
import os, shutil, argparse
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from build_atoms import (tokenize, read_sparse_mat, write_sparse_mat)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset_dir')
    ap.add_argument('--out', default=None)
    ap.add_argument('--lsa-dim', type=int, default=256)
    ap.add_argument('--levels', type=int, default=4)
    ap.add_argument('--centroids', type=int, default=32)
    ap.add_argument('--code-wt', type=float, default=0.3)
    ap.add_argument('--max-feats', type=int, default=30000)
    args = ap.parse_args()
    d = args.dataset_dir.rstrip('/')
    out = args.out or d + '-rqcodes'
    os.makedirs(out, exist_ok=True)

    trn_txt = [l.rstrip('\n') for l in open(f'{d}/raw/trn_X.txt')]
    tst_txt = [l.rstrip('\n') for l in open(f'{d}/raw/tst_X.txt')]
    lab_txt = [l.rstrip('\n') for l in open(f'{d}/raw/Y.txt')]

    vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, sublinear_tf=True,
                          max_df=0.5, min_df=3, max_features=args.max_feats,
                          token_pattern=None)
    T = vec.fit_transform(trn_txt + lab_txt)
    svd = TruncatedSVD(n_components=args.lsa_dim, random_state=0)
    E_fit = normalize(svd.fit_transform(T))
    E_tst = normalize(svd.transform(vec.transform(tst_txt)))

    print(f'residual k-means: {args.levels} levels x {args.centroids} centroids')
    codebooks, R = [], E_fit.copy()
    for l in range(args.levels):
        km = KMeans(n_clusters=args.centroids, n_init=4, random_state=l).fit(R)
        codebooks.append(km)
        R = R - km.cluster_centers_[km.labels_]
        print(f'  level {l}: residual norm {np.linalg.norm(R, axis=1).mean():.3f}')

    def encode(E):
        n = E.shape[0]
        rows, cols = [], []
        R = E.copy()
        for l, km in enumerate(codebooks):
            c = km.predict(R)
            R = R - km.cluster_centers_[c]
            rows.extend(range(n)); cols.extend(l * args.centroids + c)
        Z = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                          shape=(n, args.levels * args.centroids))
        return normalize(Z) * args.code_wt

    Z_trn, Z_tst, Z_lab = (encode(E_fit[:len(trn_txt)]), encode(E_tst),
                           encode(E_fit[len(trn_txt):]))
    names = ['#rq%dc%d' % (l, c) for l in range(args.levels)
             for c in range(args.centroids)]

    trn_X_Xf = read_sparse_mat(f'{d}/trn_X_Xf.txt')
    tst_X_Xf = read_sparse_mat(f'{d}/tst_X_Xf.txt')
    Y_Yf = read_sparse_mat(f'{d}/Y_Yf.txt')
    write_sparse_mat(sp.hstack([trn_X_Xf, Z_trn]).tocsr(), f'{out}/trn_X_Xf.txt')
    write_sparse_mat(sp.hstack([tst_X_Xf, Z_tst]).tocsr(), f'{out}/tst_X_Xf.txt')
    write_sparse_mat(sp.hstack([Y_Yf, Z_lab]).tocsr(), f'{out}/Y_Yf.txt')
    Xf = [l.rstrip('\n') for l in open(f'{d}/Xf.txt')]
    Yf = [l.rstrip('\n') for l in open(f'{d}/Yf.txt')]
    with open(f'{out}/Xf.txt', 'w') as f: f.write('\n'.join(Xf + names) + '\n')
    with open(f'{out}/Yf.txt', 'w') as f: f.write('\n'.join(Yf + names) + '\n')
    for fn in ['trn_X_Y.txt', 'tst_X_Y.txt', 'unseen_labels.txt']:
        if os.path.exists(f'{d}/{fn}'):
            shutil.copy(f'{d}/{fn}', f'{out}/{fn}')
    print('wrote', out)

if __name__ == '__main__':
    main()
