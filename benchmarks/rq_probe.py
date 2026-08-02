"""Ask what an rq codebook actually encodes, before trusting any metric computed from it.

    python benchmarks/rq_probe.py GZXML-Datasets/GZ-Reuters-90 <glove.txt>

Prints, for the labels of a dataset: how many distinct cells they occupy at each prefix
depth, which labels collide on the full tuple, what sits inside a coarse cell, and which
seen labels an unseen one is nearest to in code space.

This is the check to run FIRST. Running it late here overturned a conclusion: the codes
were being described as encoding "theme, not identity" on the strength of one example
(cotton and corn sharing 3 of 4 levels) that did not survive re-fitting -- sklearn's KMeans
is not bit-reproducible in this environment. The probe shows the opposite. On
GZ-Reuters-90 the 87 coded labels occupy 18 distinct cells at level 0 but **80-85 distinct
full 4-tuples**, and the handful of collisions are all semantically defensible (gold/silver,
gas/nat gas, cocoa/coffee, crude/veg oil). The exact count moves between runs because the
fit is not reproducible -- which is itself the reason to re-run this rather than quote it. The quantizer is fine; what was thrown away was
the conjunction, because the codes were emitted as L independent marginal features. See
``benchmarks/rq_concat.py --tuples``.
"""

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); 
import torch, numpy as np
from collections import defaultdict, Counter
from zestxml.quantize import embed_texts, rq_kmeans
from zestxml.embed import load_word_vectors
from zestxml.io import read_desc_file

D = sys.argv[1] if len(sys.argv) > 1 else "GZXML-Datasets/GZ-Reuters-90"
names = read_desc_file(f"{D}/Y.txt")
trn = read_desc_file(f"{D}/trn_X.txt")
unseen = {int(l.split()[0]) for l in open(f"{D}/unseen_labels.txt") if l.strip()}
VECTORS = sys.argv[2]
index, table = load_word_vectors(VECTORS, torch.float32)
demb, dkeep = embed_texts([" ".join(t.split()[:200]) for t in trn], index, table, require_all=False)
lemb, lkeep = embed_texts(names, index, table, require_all=True)
book, dcodes = rq_kmeans(demb, 4, 64, seed=0, log=None)
lcodes = book.transform(lemb)

print("=== 1. Do the 90 labels occupy distinct FULL tuples?")
tup = [tuple(int(c) for c in r) for r in lcodes]
print("  %d coded labels -> %d distinct full 4-tuples" % (len(tup), len(set(tup))))
for L in (1,2,3,4):
    print("    prefix of %d level(s): %d distinct cells" % (L, len({t[:L] for t in tup})))
coll = [v for v in Counter(tup).values() if v>1]
print("  labels sharing a full tuple with another: %d (groups %s)" % (sum(coll), coll))

print("\n=== 2. Labels that collide on the FULL tuple")
by = defaultdict(list)
for i,t in zip(lkeep, tup): by[t].append(names[int(i)])
for t,v in sorted(by.items(), key=lambda kv:-len(kv[1]))[:6]:
    if len(v)>1: print("  %-22s %s" % (str(t), ", ".join(v)))

print("\n=== 3. What is inside a level-0 cell? (label members)")
lvl0 = defaultdict(list)
for i,r in zip(lkeep, lcodes): lvl0[int(r[0])].append(names[int(i)])
for c,v in sorted(lvl0.items(), key=lambda kv:-len(kv[1]))[:4]:
    print("  rq0_%-3d (%2d labels): %s" % (c, len(v), ", ".join(v[:14])))

print("\n=== 4. Unseen labels: full code, and which seen labels share 3+ levels")
seen_t = {int(i):tuple(int(c) for c in r) for i,r in zip(lkeep,lcodes)}
for i in sorted(unseen)[:8]:
    if i not in seen_t: print("  %-14s NO CODE" % names[i]); continue
    t = seen_t[i]
    near = [(names[j], sum(a==b for a,b in zip(t,u))) for j,u in seen_t.items()
            if j!=i and sum(a==b for a,b in zip(t,u))>=3]
    near.sort(key=lambda x:-x[1])
    print("  %-14s %-20s shares>=3 with: %s" % (names[i], str(t),
          ", ".join("%s(%d)"%n for n in near[:6]) or "-"))
