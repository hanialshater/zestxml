"""Sweep propagation hyper-params on the held-out 20% slice of GZ-NPM TRAIN."""
import sys, itertools, numpy as np, torch
import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
torch.set_num_threads(2)
from zestxml.io import read_text_smat, read_bin_smat
from zestxml.eval import inv_propensity, evaluate
from benchmarks.baselines.label_graph import csr_to_scipy, build_graph, propagate, scipy_to_csr

S = "/tmp/claude-0/-home-user/305e430a-e06b-53e3-bf8b-d5b189305458/scratchpad/valsplit"
truth = read_text_smat(f"{S}/tst_X_Y.txt")
trn = read_text_smat(f"{S}/trn_X_Y.txt")
scores = read_bin_smat("/home/user/zestxml/Results/label_graph_val/score_mat.bin")
dt = (truth.to_dense() > 0).float()
ip = inv_propensity(trn)
freq = torch.zeros(truth.ncols).index_add_(0, trn.indices, torch.ones(trn.nnz))
unseen = freq == 0
Ys = csr_to_scipy(trn)
Sp0 = csr_to_scipy(scores)

def rep(tag, ds):
    m = evaluate(ds, dt, ip)
    mu = evaluate(ds, dt, ip, label_mask=unseen)
    ms = evaluate(ds, dt, ip, label_mask=~unseen)
    print("%-34s P@1 %6.2f P@5 %6.2f nDCG5 %6.2f PSP5 %6.2f | unseen P@1 %6.2f | seen P@1 %6.2f"
          % (tag, m["P@1"], m["P@5"], m["nDCG@5"], m["PSP@5"], mu["P@1"], ms["P@1"]), flush=True)
    return m["P@1"]

rep("baseline", scores.to_dense())
best = (None, -1)
for mode, k in [("cosine", 5), ("cosine", 15), ("cosine", 50), ("ppmi", 15)]:
    G = build_graph(Ys, k=k, mode=mode)
    for a, steps, restrict in itertools.product([0.05, 0.1, 0.2, 0.35, 0.5], [1, 2], [1, 0]):
        Sn = propagate(Sp0, G, a, steps, bool(restrict))
        p1 = rep(f"{mode} k={k} a={a} steps={steps} restrict={restrict}",
                 scipy_to_csr(Sn).to_dense())
        if p1 > best[1]:
            best = ((mode, k, a, steps, restrict), p1)
print("BEST", best)
