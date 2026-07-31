"""Make an 80/20 point split of GZ-NPM train for tuning the propagation weight a."""
import os, sys, numpy as np, torch
sys.path.insert(0, "/home/user/zestxml")
torch.set_num_threads(2)
from zestxml.io import read_text_smat, write_text_smat

D = "/home/user/zestxml/GZXML-Datasets/GZ-NPM"
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
X = read_text_smat(f"{D}/trn_X_Xf.txt")
Y = read_text_smat(f"{D}/trn_X_Y.txt")
n = X.nrows
rng = np.random.RandomState(0)
perm = rng.permutation(n)
va = np.sort(perm[: n // 5]); tr = np.sort(perm[n // 5:])
for name, idx in (("trn", tr), ("tst", va)):
    t = torch.as_tensor(idx)
    write_text_smat(X.select_rows(t), f"{OUT}/{name}_X_Xf.txt")
    write_text_smat(Y.select_rows(t), f"{OUT}/{name}_X_Y.txt")
print("split done", len(tr), len(va))
