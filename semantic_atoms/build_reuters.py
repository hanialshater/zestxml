"""Build a GZXML-format zero-shot dataset from Reuters-21578 (via NLTK).

Produces GZXML-Datasets/GZ-Reuters-90 with the exact file layout ZestXML
expects (Xf.txt, Yf.txt, trn/tst_X_Xf.txt, Y_Yf.txt, trn/tst_X_Y.txt) plus
raw/ text files used by build_atoms.py, and unseen_labels.txt marking the
labels held out from training (zero training column => ZestXML's own
remove_test_labels treats them as unseen).
"""
import os, re, sys, argparse
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nltk
from nltk.corpus import reuters, stopwords
from nltk.stem import PorterStemmer

STOPWORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()
TOKEN_RE = re.compile(r"[a-z][a-z0-9]+")

# Reuters-21578 category codes -> descriptive label text (standard expansions
# of the Reuters topic codes; applied identically for all compared models).
LABEL_TEXT = {
    'acq': 'acquisitions mergers takeovers', 'alum': 'aluminium aluminum',
    'barley': 'barley',
    'bop': 'balance of payments', 'carcass': 'carcass meat',
    'castor-oil': 'castor oil', 'cocoa': 'cocoa', 'coconut': 'coconut',
    'coconut-oil': 'coconut oil', 'coffee': 'coffee', 'copper': 'copper',
    'copra-cake': 'copra cake', 'corn': 'corn maize', 'cotton': 'cotton',
    'cotton-oil': 'cottonseed oil', 'cpi': 'consumer price index inflation',
    'cpu': 'capacity utilisation utilization', 'crude': 'crude oil petroleum',
    'dfl': 'dutch guilder florin', 'dlr': 'dollar', 'dmk': 'german mark deutsche mark',
    'earn': 'earnings corporate profits', 'fuel': 'fuel oil', 'gas': 'gasoline',
    'gnp': 'gross national product gross domestic product',
    'gold': 'gold', 'grain': 'grain cereals', 'groundnut': 'groundnut peanut',
    'groundnut-oil': 'groundnut peanut oil', 'heat': 'heating oil',
    'hog': 'hogs pigs pork', 'housing': 'housing starts construction',
    'income': 'personal income', 'instal-debt': 'installment debt consumer credit',
    'interest': 'interest rates', 'ipi': 'industrial production index',
    'iron-steel': 'iron steel', 'jet': 'jet fuel kerosene',
    'jobs': 'jobs employment unemployment', 'l-cattle': 'live cattle',
    'lead': 'lead metal', 'lei': 'leading economic indicators',
    'lin-oil': 'linseed oil', 'livestock': 'livestock', 'lumber': 'lumber timber wood',
    'meal-feed': 'meal feed animal feedstuff', 'money-fx': 'money foreign exchange currency',
    'money-supply': 'money supply', 'naphtha': 'naphtha', 'nat-gas': 'natural gas',
    'nickel': 'nickel', 'nkr': 'norwegian krone', 'nzdlr': 'new zealand dollar',
    'oat': 'oats', 'oilseed': 'oilseeds', 'orange': 'orange juice citrus',
    'palladium': 'palladium', 'palm-oil': 'palm oil', 'palmkernel': 'palm kernel',
    'pet-chem': 'petrochemicals chemicals', 'platinum': 'platinum',
    'potato': 'potatoes', 'propane': 'propane', 'rand': 'south african rand',
    'rape-oil': 'rapeseed oil', 'rapeseed': 'rapeseed', 'reserves': 'foreign exchange reserves',
    'retail': 'retail sales', 'rice': 'rice', 'rubber': 'rubber', 'rye': 'rye',
    'ship': 'shipping ships vessels tanker', 'silver': 'silver',
    'sorghum': 'sorghum', 'soy-meal': 'soybean meal', 'soy-oil': 'soybean oil',
    'soybean': 'soybeans', 'strategic-metal': 'strategic metals',
    'sugar': 'sugar', 'sun-meal': 'sunflower meal', 'sun-oil': 'sunflower oil',
    'sunseed': 'sunflower seed', 'tea': 'tea', 'tin': 'tin',
    'trade': 'trade balance deficit surplus', 'veg-oil': 'vegetable oil',
    'wheat': 'wheat', 'wpi': 'wholesale price index', 'yen': 'japanese yen',
    'zinc': 'zinc',
}

def tokenize(text):
    toks = TOKEN_RE.findall(text.lower())
    return [STEMMER.stem(t) for t in toks if t not in STOPWORDS]

def write_sparse_mat(X, filename):
    X = X.tocsr(); X.sort_indices()
    with open(filename, 'w') as f:
        print("%d %d" % X.shape, file=f)
        for i in range(X.shape[0]):
            r = X.getrow(i)
            print(' '.join('%d:%.5f' % (j, v) for j, v in zip(r.indices, r.data)), file=f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='GZXML-Datasets/GZ-Reuters-90')
    ap.add_argument('--num-unseen', type=int, default=25)
    ap.add_argument('--max-feats', type=int, default=20000)
    args = ap.parse_args()
    os.makedirs(f'{args.out}/raw', exist_ok=True)

    labels = sorted(reuters.categories())
    assert all(l in LABEL_TEXT for l in labels), 'missing label expansion'
    lab_idx = {l: i for i, l in enumerate(labels)}

    trn_ids = sorted(i for i in reuters.fileids() if i.startswith('training'))
    tst_ids = sorted(i for i in reuters.fileids() if i.startswith('test'))
    clean = lambda t: re.sub(r'\s+', ' ', t).strip()
    trn_txt = [clean(reuters.raw(i)) for i in trn_ids]
    tst_txt = [clean(reuters.raw(i)) for i in tst_ids]
    lab_txt = [f"{l.replace('-', ' ')} {LABEL_TEXT[l]}" for l in labels]

    def label_mat(ids):
        M = sp.lil_matrix((len(ids), len(labels)), dtype=np.float32)
        for r, fid in enumerate(ids):
            for c in reuters.categories(fid):
                M[r, lab_idx[c]] = 1.0
        return M.tocsr()
    trn_X_Y, tst_X_Y = label_mat(trn_ids), label_mat(tst_ids)

    # --- zero-shot split: hold out labels spread across the frequency range ---
    trn_freq = np.asarray(trn_X_Y.sum(0)).ravel()
    tst_freq = np.asarray(tst_X_Y.sum(0)).ravel()
    cand = [i for i in np.argsort(trn_freq) if tst_freq[i] >= 5 and trn_freq[i] < 300]
    step = max(1, len(cand) // args.num_unseen)
    unseen = sorted(cand[::step][:args.num_unseen])
    print('unseen labels:', [labels[i] for i in unseen])

    keep = np.ones(len(labels), bool); keep[unseen] = False
    mask = sp.diags(keep.astype(np.float32))
    trn_X_Y = (trn_X_Y @ mask).tocsr(); trn_X_Y.eliminate_zeros()
    nz = np.asarray(trn_X_Y.sum(1)).ravel() > 0   # drop train docs left label-less
    print(f'dropping {np.sum(~nz)} train docs with only unseen labels')
    trn_X_Y = trn_X_Y[nz]
    trn_txt = [t for t, k in zip(trn_txt, nz) if k]

    # --- document features: stemmed unigrams + bigrams, tf-idf ---
    uni = TfidfVectorizer(tokenizer=tokenize, lowercase=False, ngram_range=(1, 1),
                          max_df=0.5, min_df=2, sublinear_tf=True,
                          max_features=args.max_feats, token_pattern=None)
    big = TfidfVectorizer(tokenizer=tokenize, lowercase=False, ngram_range=(2, 2),
                          max_df=0.5, min_df=3, sublinear_tf=True,
                          max_features=args.max_feats, token_pattern=None)
    trn_U = uni.fit_transform(trn_txt); trn_B = big.fit_transform(trn_txt)
    tst_U, tst_B = uni.transform(tst_txt), big.transform(tst_txt)
    Xf = list(uni.get_feature_names_out()) + [' '.join(b.split()) for b in big.get_feature_names_out()]
    trn_X_Xf = normalize(sp.hstack([trn_U, trn_B]).tocsr())
    tst_X_Xf = normalize(sp.hstack([tst_U, tst_B]).tocsr())

    # --- label features: same text vectorizers + per-label unique feature ---
    lab_U, lab_B = uni.transform(lab_txt), big.transform(lab_txt)
    lab_text_feats = normalize(sp.hstack([lab_U, lab_B]).tocsr())
    uniq = sp.identity(len(labels), format='csr', dtype=np.float32)
    Y_Yf = normalize(sp.hstack([lab_text_feats, uniq]).tocsr())
    Yf = Xf + ['__label__%d__%s' % (i, labels[i]) for i in range(len(labels))]

    d = args.out
    write_sparse_mat(trn_X_Xf, f'{d}/trn_X_Xf.txt')
    write_sparse_mat(tst_X_Xf, f'{d}/tst_X_Xf.txt')
    write_sparse_mat(Y_Yf, f'{d}/Y_Yf.txt')
    write_sparse_mat(trn_X_Y, f'{d}/trn_X_Y.txt')
    write_sparse_mat(tst_X_Y, f'{d}/tst_X_Y.txt')
    with open(f'{d}/Xf.txt', 'w') as f: f.write('\n'.join(Xf) + '\n')
    with open(f'{d}/Yf.txt', 'w') as f: f.write('\n'.join(Yf) + '\n')
    with open(f'{d}/raw/trn_X.txt', 'w') as f: f.write('\n'.join(trn_txt) + '\n')
    with open(f'{d}/raw/tst_X.txt', 'w') as f: f.write('\n'.join(tst_txt) + '\n')
    with open(f'{d}/raw/Y.txt', 'w') as f: f.write('\n'.join(lab_txt) + '\n')
    with open(f'{d}/unseen_labels.txt', 'w') as f:
        f.write('\n'.join(map(str, unseen)) + '\n')
    print(f'wrote {d}: trn {trn_X_Xf.shape}, tst {tst_X_Xf.shape}, Y {Y_Yf.shape}, '
          f'{len(unseen)} unseen labels')

if __name__ == '__main__':
    main()
