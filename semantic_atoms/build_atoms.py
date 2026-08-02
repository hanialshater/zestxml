"""Sparse semantic atoms for ZestXML — dataset-agnostic.

Reads any GZXML-format dataset dir that has raw/{trn_X,tst_X,Y}.txt, learns a
sparse semantic basis SHARED by documents and labels, and writes an augmented
copy <dir>-atoms where the atom features are concatenated to the existing
lexical tf-idf features in Xf/Yf (and the corresponding matrices).

Pipeline: corpus tf-idf -> LSA (TruncatedSVD) -> k-sparse autoencoder (tied
weights, top-k activation, unit-norm decoder atoms) -> top-k atom activations
as features. Atom feature tokens are '#atom<j>' on BOTH sides — no underscore,
so ZestXML's direct Xf<->Yf map (helper.cpp:create_Xf_Yf_map_direct, which
strips through the first '_' of a Yf token) matches them exactly.

Nothing external: the semantic basis is learned from the training corpus and
the label texts only (both available at train time in the GZXML setting; test
documents are only ever transformed).
"""
import os, re, sys, shutil, argparse
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

STOPWORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()
TOKEN_RE = re.compile(r"[a-z][a-z0-9]+")

def tokenize(text):
    toks = TOKEN_RE.findall(text.lower())
    return [STEMMER.stem(t) for t in toks if t not in STOPWORDS]

def read_sparse_mat(filename):
    with open(filename) as f:
        nr, nc = map(int, f.readline().split())
        data, indices, indptr = [], [], [0]
        for line in f:
            row = line.split()
            for tok in row:
                j, v = tok.split(':')
                indices.append(int(j)); data.append(float(v))
            indptr.append(len(data))
    return sp.csr_matrix((data, indices, indptr), shape=(nr, nc), dtype=np.float32)

def write_sparse_mat(X, filename):
    X = X.tocsr(); X.sort_indices()
    with open(filename, 'w') as f:
        print("%d %d" % X.shape, file=f)
        for i in range(X.shape[0]):
            r = X.getrow(i)
            print(' '.join('%d:%.5f' % (j, v) for j, v in zip(r.indices, r.data)), file=f)

class KSparseAE(torch.nn.Module):
    """Tied-weight k-sparse autoencoder with unit-norm decoder atoms."""
    def __init__(self, d_in, n_atoms, k):
        super().__init__()
        self.k = k
        self.W = torch.nn.Parameter(torch.randn(d_in, n_atoms) * (1.0 / d_in ** 0.5))
        self.b = torch.nn.Parameter(torch.zeros(n_atoms))

    def normalize_atoms(self):
        with torch.no_grad():
            self.W /= self.W.norm(dim=0, keepdim=True).clamp_min(1e-8)

    def encode(self, x):
        z = torch.relu(x @ self.W + self.b)
        if self.k < z.shape[1]:
            thresh = z.topk(self.k, dim=1).values[:, -1:]
            z = z * (z >= thresh)
        return z

    def forward(self, x):
        z = self.encode(x)
        return z @ self.W.T, z

def train_sae(E, n_atoms, k, epochs=40, bs=512, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    model = KSparseAE(E.shape[1], n_atoms, k)
    model.normalize_atoms()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X = torch.from_numpy(E.astype(np.float32))
    for ep in range(epochs):
        perm = torch.randperm(len(X))
        tot = 0.0
        for i in range(0, len(X), bs):
            xb = X[perm[i:i + bs]]
            recon, z = model(xb)
            loss = ((recon - xb) ** 2).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            model.normalize_atoms()
            tot += loss.item() * len(xb)
        if ep % 10 == 0 or ep == epochs - 1:
            print(f'  sae epoch {ep}: recon mse {tot / len(X):.4f}')
    return model

@torch.no_grad()
def encode_sparse(model, E, bs=2048):
    rows = []
    for i in range(0, len(E), bs):
        z = model.encode(torch.from_numpy(E[i:i + bs].astype(np.float32)))
        rows.append(sp.csr_matrix(z.numpy()))
    Z = sp.vstack(rows).tocsr()
    Z.eliminate_zeros()
    return normalize(Z)   # unit-norm atom block per row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset_dir')
    ap.add_argument('--out', default=None, help='default: <dataset_dir>-atoms')
    ap.add_argument('--lsa-dim', type=int, default=256)
    ap.add_argument('--n-atoms', type=int, default=4096)
    ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--atom-wt', type=float, default=1.0,
                    help='weight of the (unit-norm) atom block relative to the lexical block')
    ap.add_argument('--max-feats', type=int, default=30000)
    ap.add_argument('--epochs', type=int, default=40)
    args = ap.parse_args()
    d = args.dataset_dir.rstrip('/')
    out = args.out or d + '-atoms'
    os.makedirs(out, exist_ok=True)

    trn_txt = [l.rstrip('\n') for l in open(f'{d}/raw/trn_X.txt')]
    tst_txt = [l.rstrip('\n') for l in open(f'{d}/raw/tst_X.txt')]
    lab_txt = [l.rstrip('\n') for l in open(f'{d}/raw/Y.txt')]

    print('fitting shared tf-idf + LSA on train docs + label texts...')
    vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, sublinear_tf=True,
                          max_df=0.5, min_df=3, max_features=args.max_feats,
                          token_pattern=None)
    T_trn = vec.fit_transform(trn_txt + lab_txt)
    svd = TruncatedSVD(n_components=args.lsa_dim, random_state=0)
    E_fit = normalize(svd.fit_transform(T_trn))
    print(f'  LSA explained variance: {svd.explained_variance_ratio_.sum():.3f}')

    print(f'training k-sparse AE ({args.n_atoms} atoms, k={args.k})...')
    model = train_sae(E_fit, args.n_atoms, args.k, epochs=args.epochs)

    E_trn = E_fit[:len(trn_txt)]
    E_lab = E_fit[len(trn_txt):]
    E_tst = normalize(svd.transform(vec.transform(tst_txt)))
    Z_trn = encode_sparse(model, E_trn) * args.atom_wt
    Z_tst = encode_sparse(model, E_tst) * args.atom_wt
    Z_lab = encode_sparse(model, E_lab) * args.atom_wt

    used = np.union1d(np.union1d(Z_trn.indices, Z_tst.indices), Z_lab.indices)
    remap = sp.csr_matrix((np.ones(len(used), np.float32),
                           (used, np.arange(len(used)))),
                          shape=(args.n_atoms, len(used)))
    Z_trn, Z_tst, Z_lab = Z_trn @ remap, Z_tst @ remap, Z_lab @ remap
    atom_names = ['#atom%d' % j for j in used]
    print(f'{len(used)} atoms in use; mean active/doc: {Z_trn.getnnz(1).mean():.1f}, '
          f'/label: {Z_lab.getnnz(1).mean():.1f}')

    print('writing augmented dataset to', out)
    trn_X_Xf = read_sparse_mat(f'{d}/trn_X_Xf.txt')
    tst_X_Xf = read_sparse_mat(f'{d}/tst_X_Xf.txt')
    Y_Yf = read_sparse_mat(f'{d}/Y_Yf.txt')
    write_sparse_mat(sp.hstack([trn_X_Xf, Z_trn]).tocsr(), f'{out}/trn_X_Xf.txt')
    write_sparse_mat(sp.hstack([tst_X_Xf, Z_tst]).tocsr(), f'{out}/tst_X_Xf.txt')
    write_sparse_mat(sp.hstack([Y_Yf, Z_lab]).tocsr(), f'{out}/Y_Yf.txt')
    Xf = [l.rstrip('\n') for l in open(f'{d}/Xf.txt')]
    Yf = [l.rstrip('\n') for l in open(f'{d}/Yf.txt')]
    with open(f'{out}/Xf.txt', 'w') as f: f.write('\n'.join(Xf + atom_names) + '\n')
    with open(f'{out}/Yf.txt', 'w') as f: f.write('\n'.join(Yf + atom_names) + '\n')
    for fn in ['trn_X_Y.txt', 'tst_X_Y.txt', 'unseen_labels.txt']:
        if os.path.exists(f'{d}/{fn}'):
            shutil.copy(f'{d}/{fn}', f'{out}/{fn}')
    print('done')

if __name__ == '__main__':
    main()
