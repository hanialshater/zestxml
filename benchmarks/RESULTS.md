# ZestXML: results

Section 0 is the synthesis of everything measured. Everything after it is the
chronological record, kept intact including the runs whose conclusions were later
reversed -- the reversals are listed in 0.5.

---

# 0. Summary: what was tested, what held, what did not

Everything below was measured in this repository. Sections after this one are the
chronological record, including the runs that produced conclusions later reversed; this is
the synthesis. Where a number here disagrees with prose further down, this section wins and
the reversal is named explicitly under *Conclusions that were wrong*.

Two bodies of work:

1. **XMC**, on GZ-NPM (3223 labels, 286 unseen) and GZ-Reuters-90 (90 labels, 15 unseen),
   plus one standard benchmark, AmazonCat-13K.
2. **ZestXML against SASRec**, on ml-1m and Amazon Video_Games, framed as XMC.

---

## 1. The port

The C++ implementation is reimplemented in PyTorch (`zestxml/`), with a clear API
(`ZestXML(data_dir, res_dir, **options)`), CSR primitives on torch tensors, and GPU support.
It reproduces the reference numbers on GZ-NPM exactly (P@1 73.04, unseen 52.11).

On **AmazonCat-13K** (1.19M train, 306K test, 13330 labels, no zero-shot split):

| run | pattern nnz | test shortlist recall | P@1 | P@5 | PSP@5 |
|---|---|---|---|---|---|
| title only | 11.2M | 82.05 | 76.53 | — | — |
| title + `--content` | 270.4M | 92.46 | **93.41** | 64.29 | 72.15 |

The title-only run was mis-specified rather than weak: AmazonCat-13K's published numbers use
the product description, and `-Titles-` is a *different* dataset. Feeding the description
lifts P@1 by 16.9 points. **The comparison band remains unverified** — the figures
circulating for this dataset came from a web search, not a paper anyone here read, and every
source that would settle it is unreachable from this sandbox. 93.41 is plausible for a
sparse linear model against transformer methods; plausible is not checked.

Training is not bit-reproducible above `num_thread=1` (torch CPU reduction order); metric
noise is about 0.011, up to 6e-2 in raw bilinear scores. Treat differences below ~0.1 P@1 as
nothing.

## 2. XMC: the two things that worked

**Pattern pruning — the largest overall gain measured anywhere here.** `-prune_vectors` /
`-prune_min_sim` drop mined `(xf, yf)` pairs whose feature names are semantically unrelated.
On GZ-NPM, keeping 73% of the pattern:

| min_sim | P@1 | seen P@1 | unseen P@1 |
|---|---|---|---|
| 0.00 | 73.02 | 74.95 | 52.07 |
| **0.30** | **74.57** | **76.56** | 52.13 |

**+1.55 P@1, +1.61 seen, unseen unchanged** — so it makes the model better, not more
zero-shot. It loses on Reuters, and the reason is scale: `bs_count` keeps the top-k label
features per point feature by co-occurrence alone, and on a large sparse label space many of
those slots go to pairs that co-occur without being related. Pruning frees budget spent on
noise. Reuters mines 284K entries over 90 labels and the budget never binds. The sweep had
not found its optimum at 0.30.

**Label feature-bag expansion — the only large zero-shot gain, and its sign is predictable.**
Widening each label's feature bag with GloVe neighbours (k=2, cosine floor 0.7):

| dataset | unseen P@1 | delta |
|---|---|---|
| GZ-Reuters-90 | 61.28 -> **69.74** | **+8.5** |
| GZ-NPM | — | **-7.3** |

The sign is decided by how widely the added features are *shared*, not by coverage. A
neighbour token that lands on a handful of labels sharpens; one that lands on hundreds
blurs. GloVe beat MiniLM by 6.4 points of unseen P@1 at the same settings and CLIP was worse
than no expansion at all — almost certainly a *threshold* result rather than an encoder
result, since transformer embedding spaces are anisotropic and a cosine floor tuned on GloVe
is permissive on MiniLM.

## 3. XMC: what did not work, and why

**Retrieval is not the binding constraint.** Confirmed four independent ways. Handing the
model 12.5–15 extra points of unseen shortlist recall produced *zero* unseen P@1, including
after closing the train/test mismatch by regenerating both shortlists and retraining from
scratch. The explanation consistent with every arm: 99.09% of npm's and 100% of Reuters'
unseen-label tokens are already in `Xf`, so an exact lexical link already fires for
essentially every unseen label. A semantic channel can only add a fuzzier version of a link
that already exists in its sharpest form, and the labels it newly retrieves are ones the
exact channel already considered and correctly declined.

**RQ-KMeans semantic IDs are neutral-to-negative in every form tried** — as peer features at
any mass, as prefix tuples, as multi-code assignments, as concatenated blocks, and with
transformer encoders on both sides. Two mechanisms, both measured: marginals are too coarse
to rank, tuples are too sparse to fire (+2.5 recall against marginals' +10.1). More code
mass monotonically costs unseen accuracy.

**Structured hard pursuit over semantic-ID prefix blocks** (`zestxml/pursuit.py`): block
masking with ancestor closure and gradient-driven revival, implemented properly and working
mechanically. On GZ-NPM it beats the unpruned control but **loses to the one-shot cosine
prune already in the repo** (73.72 against 74.59) *at lower sparsity*, so the selection is
worse, not the budget. The tell that it is not the structure doing the work: the degenerate
`xf x prefix` mode — 257634 blocks over 284540 slots, 1.1 each, where block pruning and
entry pruning are the same operation — scored best of the pursuit arms.

**Per-label classifiers score exactly zero on unseen labels**, as they must. BM25 with no
training at all matches ZestXML's unseen P@1 on Reuters.

**EMMETT / IRENE** (KDD'24, synthesizing an unseen label's classifier from similar seen
ones): the mechanism works — one-vs-all goes from the tie-break floor to 27.96 unseen P@1
through the learned generator, and the generator earns +13.1 PSP@5 over mean synthesis on
the tail. But **synthesis does not beat the trivial baseline on unseen labels**: keeping
classifiers where they exist and falling back to the encoder where they do not reaches
42.19, above the full method's 41.25. All the unseen accuracy comes from the encoder term.
What synthesis buys is head and tail accuracy.

## 4. ZestXML against SASRec

Framed as XMC: a point is a user, its labels are the set of items touched in the next
`horizon` steps, metrics are P@k / nDCG@k / PSP@k. SASRec keeps its next-item training
signal; only the evaluation is shared. `sasrec+content` adds a learned linear map from the
*same* tf-idf text ZestXML reads, so the cold column is a baseline and not a strawman.
`random` and `popularity` rows are mandatory: group metrics mask labels, so the cold row
asks "rank the N cold items" and has a floor well above zero.

**Amazon Video_Games** (8/20-core, 2885 items, 4000 evaluated, 5% cold):

| model | all P@1 | head P@1 | tail P@1 | **cold P@1** | **cold PSP@5** |
|---|---|---|---|---|---|
| random | 0.10 | 0.32 | 0.00 | 0.46 | 2.60 |
| popularity | 1.70 | 2.20 | 0.09 | 0.31 | 3.03 |
| zestxml | 3.05 | 4.31 | 2.04 | **4.92** | **14.31** |
| **sasrec** | **4.10** | **5.67** | 2.57 | 0.46 | 3.18 |
| sasrec+content | 3.88 | 5.57 | **2.78** | 0.77 | 4.77 |

**The trade, demonstrated.** SASRec wins every column with training signal behind it.
ZestXML wins cold by **6.4x on P@1 and 3.0x on PSP@5** against the strongest baseline, and
10.7x against chance. Plain SASRec sits exactly at the random floor there, which is what a
model with untrained embeddings for those items should do.

**Content projection is not a substitute for scoring through the label's own words.**
`sasrec+content` buys the tail (2.78, the best tail number in the table) and almost nothing
on cold (0.77 against 4.92). A map trained on warm items transfers to rare ones; it does not
manufacture a representation for an item the encoder has never seen.

**The deciding variable is text discriminativeness, and it is measurable.** `profile()`
reports, for the average item, how many other items share its title tokens:

| dataset | items | interactions/item | tokens shared with | cold outcome |
|---|---|---|---|---|
| ml-1m | 3706 | 255.6 | **448.8** | everything at the floor |
| Amazon (first, broken) | 13722 | 1.6 | **3812.3** | everything at the floor |
| Amazon Video_Games | 2885 | 49.1 | **151.4** | ZestXML 6.4x |

On ml-1m the average movie shares its title tokens with 449 others — three genre words and a
name — so the feature bag carries no identity and only the per-label identity feature
discriminates, which is exactly what a cold item lacks. **Read this line before reading any
table.** Interactions per item in the tens, under-10 share below ~20%, token sharing in the
tens rather than the hundreds.

## 5. Conclusions that were wrong, and the corrections

Recorded because each one survived a first glance.

* **"ZestXML owns the cold column" (ml-1m, next-item framing).** Plain SASRec scored exactly
  0.00 there against ZestXML's 1.34, which looked decisive. Under XMC group-masked metrics
  the same run gives ZestXML 1.07 and `sasrec+content` 1.91 — and the random floor is 0.31,
  so **both were within ~3x of chance**. The 0.00 was an artifact of asking "surface a cold
  item against 3706", which has a floor of zero for untrained embeddings. Corrected twice: to
  "SASRec wins cold", then to "nobody wins cold on ml-1m".
* **"Codes encode theme, not identity."** Recorded from one example (`cotton`/`corn` sharing
  3 of 4 levels). Did not survive re-fitting — sklearn's KMeans is not bit-reproducible here
  — and a proper probe said close to the opposite: 87 labels occupy 80–85 distinct 4-tuples.
* **"The identity is in the conjunction, so emit prefix tuples."** Followed from the probe
  and was refuted by measurement, not argument: tuples fire far less often than marginals
  and more mass makes it worse.
* **"Co-occurrence was already choosing the right pairs."** Drawn from Reuters alone. Wrong
  on any label space large enough for the pattern budget to bind — on npm, pruning is the
  biggest win in the repo.
* **Three unverifiable download URLs shipped** (a guessed Drive id, an archive.org pattern, a
  dataset absent from the mirror), and a reference band printed next to the wrong dataset.
  Both now marked unverified rather than presented as fact.
* **A stale `seen_labels.txt`** shared between datasets made a whole npm run report P@1 52.15
  against a true 73.04 — Reuters' 75 ids are valid indices into npm's 3223, so nothing
  complained, and shortlist recall matched the reference *exactly*. The pipeline now rejects
  a seen-labels cache that disagrees with `trn_X_Y`.
* **The first Amazon run was uninterpretable** for four harness reasons at once: catalogue
  derived from users read rather than users that survived the length filter, no k-core,
  generic category tokens appended to every title, and a cold fraction that swallowed 73% of
  the test signal. Every arm sat at the noise floor, which reads as a result.

The pattern in all of these: a number that looks like a finding is usually a harness
property. The controls that caught them — `random`, `popularity`, `profile()`, shortlist
recall, derived-set validation — cost little and are now default.

## 6. What this implies for interest mining at scale

The target setting is roughly 10M customers, 15k interests, ~2% labelled (~133 positives per
label), with an open and growing interest vocabulary.

* **Neither method alone is right.** SASRec-style scoring wins wherever an interest has
  history; ZestXML's text channel wins where it does not, by 6.4x, and the open vocabulary
  means the cold case is not a rare corner but continuous inflow.
* **Fuse them, and tune the fusion on a label-holdout split.** An earlier attempt at score
  fusion failed for a structural reason — weights tuned on a normal validation split cannot
  see the unseen regime they are meant to govern.
* **Expect the text channel to matter more than these numbers suggest.** Interest names are
  short, meaningful and comparatively discriminative; 133 positives per label is far sparser
  than Amazon Video_Games' 49 interactions per item.
* **Spend effort on the pattern, not on semantic IDs.** Pruning is worth +1.55 P@1 and the
  sweep was still climbing. Every semantic-ID variant tried — eight of them — was
  neutral-to-negative.
* **Measure `profile()` on your own data first.** If interest names share their tokens with
  hundreds of other interests, the text channel will not fire and none of the cold-start
  result transfers.

Open and worth doing, in order: raise `shorty_k` until test shortlist recall plateaus (it was
48.61% on Amazon and 48.88% on ml-1m, so the warm columns are retrieval-capped and the
overall gap is partly artificial); a label-holdout fusion of the two scorers; the prune sweep
past min_sim 0.30 on npm; and a verified AmazonCat-13K reference table.

---

# Baseline sweep vs. the ZestXML PyTorch port (GZ-NPM, GZ-Reuters-90)

Six experiment agents, all reporting `status: ok`. Every headline number below was
**re-measured by me** with `tools/eval_xc.py` against the score matrices on disk, not
copied from the agents' reports. Where an agent's prose disagreed with the artifact, the
artifact wins and the discrepancy is called out.

Reference = ZestXML PyTorch port, exact direct map. The canonical artifact is
`/home/user/zestxml/Results/Npm2exact/score_mat.bin`, which reproduces the briefing
numbers exactly (73.04 / 40.69 / 60.13 / 19.10 / 28.62, unseen 52.11).

> **Stale reference matrix, read this before comparing seen-only numbers.**
> `/home/user/zestxml/Results/GZ-NPM-torch/score_mat.bin` is an *older* run of the same
> config and evaluates to P@1 **71.99**, PSP@1 18.70, unseen 51.73, seen-only P@1 73.85 /
> seen PSP@1 46.85. The `ova_linear` agent used that file for all its seen-only ZestXML
> comparisons. The correct reference seen-only figures (from `Npm2exact`) are
> **P@1 74.97, PSP@1 47.91, PSP@5 57.64**. This makes the OVA gap slightly *larger* than
> that agent reported; its conclusion is unaffected but its numbers are off.

---

## 1. Comparison table — GZ-NPM (8376 test points, 3223 labels, 286 unseen)

Sorted by unseen-only P@1, because that is what this benchmark exists to measure.
All values re-verified by me. ✗ in the "reaches unseen?" column = the method has **no
mechanism at all** to score a label with no training positive.

| Method | reaches unseen? | P@1 | P@5 | PSP@1 | PSP@5 | unseen P@1 | recall@100 |
|---|---|---|---|---|---|---|---|
| **ZestXML reference** (`Npm2exact`) | ✓ | **73.04** | 40.69 | 19.10 | 28.62 | **52.11** | 71.8 |
| lowrank, r=64 primary (`lowrank`) | ✓ | 67.54 | 38.13 | 18.52 | 29.63 | 53.16 ⚠ | 71.82 |
| label-graph propagation (`label_graph`) | ✓ (no effect) | 73.09 | **41.23** | **19.24** | 30.19 | 52.11 (identical) | 71.8 |
| better vectors, fastText fallback (`better_vectors`) | ✓ | 73.07 | 40.54 | 19.13 | 28.31 | 50.86 | 71.80 |
| classical hybrid kNN+BM25 (`classical`) | ✓ (via BM25) | 71.30 | 39.62 | 18.81 | **36.94** | 47.45 | 78.61 † |
| BM25 only, zero training (`classical-bm25`) | ✓ | 38.35 | 31.45 | **21.95** | **38.40** | 47.45 | 57.43 |
| dense probe, Numberbatch (`dense_probe`) | ✓ | 37.71 | 18.23 | 20.27 | 22.42 | 33.20 | 39.32 |
| SPLADE-style learned sparse (`splade`) | ✓ (nominally) | 70.88 | 38.01 | 16.95 | 24.05 | 23.00 | n/a ‡ |
| OVA linear, 1-vs-all (`ova_linear`) | **✗** | 71.18 | 39.20 | 16.02 | 23.31 | 2.49 = **0** | n/a ‡ |
| kNN k=25 (`classical-knn25`) | **✗** | 70.63 | 37.09 | 16.89 | 22.78 | 2.49 = **0** | 62.98 |
| tf-idf centroid (`classical-centroid`) | **✗** | 66.12 | 35.58 | 21.80 | 24.91 | 2.49 = **0** | 65.82 |
| one-hot lexical control (`dense_probe_onehot`) | ✓ | — | — | — | — | — | 47.16 |
| ZestXML lexical shortlist (candidate gen. only) | ✓ | — | — | — | — | — | 71.82 |

† not comparable: the hybrid union uses ~200 candidates, everything else 100.
‡ the method scores all 3223 labels directly and has no separate shortlisting stage, so a
shortlist recall figure has no counterpart; neither agent computed a top-100 recall.
recall@100 is the **micro** definition (total hits / total true), the one that reproduces
the quoted 71.8%. Macro (mean per-point) recall of the same shortlist is 80.81 — do not
mix the two.

**The 2.49 is not a score.** I checked the artifacts directly: `ova_linear`,
`classical-knn25` and `classical-centroid` score matrices contain **exactly zero non-zero
entries in any of the 286 unseen columns**. The evaluator masks seen labels to `-inf`,
then top-k's a row of all-zero unseen scores and breaks the tie by label id. 2.49 is the
tie-break floor of this dataset, nothing else. Read it as 0.

This specifically corrects the `ova_linear` agent, which wrote that its unseen P@1 is
"data-dependent noise, not exactly zero", pointing at non-zero weights in the unseen
columns of its dense `W`. That is true of `W` but irrelevant to the submitted artifact:
the matrix is truncated to top-100 labels per point and no unseen label ever survives, so
the stored unseen scores are all implicit zeros. It lands on precisely the same 2.49 as
kNN and centroid, which is the giveaway.

### GZ-Reuters-90 (3019 points, 90 labels, 15 unseen)

Only 3 of 6 methods ran Reuters at all. Missing entries are **not measured**, not zero.

| Method | P@1 | unseen P@1 |
|---|---|---|
| ZestXML reference (`Reu2exact`) | 86.35 | 61.28 |
| better vectors, fastText fallback | 86.25 | **64.47** |
| kNN k=25 | 81.05 | 0 (2.07 artifact) |
| OVA linear | 81.78 | 0 (2.07 artifact) |
| BM25 only, zero training | 18.18 | **61.28** |
| classical hybrid | 72.84 | 61.28 |
| splade / dense probe / lowrank / label-graph | **not run** | **not run** |

---

## 2. Per-method notes

### better_vectors — fuzzy direct map with fastText / Numberbatch (9 min)
Replaces GloVe-100 in ZestXML's `-direct_map vectors` fuzzy label-name matching with
fastText wiki-news-subwords-300 and ConceptNet Numberbatch, in both fallback and augment
mode; 4 NPM runs + 2 Reuters runs, reference hyperparameters throughout.
**Result: ties the reference on all-label metrics (73.07 vs 73.04) and loses on the metric
it targeted (unseen P@1 50.86 vs 52.11).** Wins on Reuters (unseen 64.47 vs 61.28).

The agent's diagnosis is the strongest single piece of analysis in the sweep and I accept
it. Three independent legs: (a) better vectors buy no coverage on npm — the distributed
fastText *text table* is subword-trained but not subword-queryable, so npm label-token
coverage moves 77.1% → 77.8% and Numberbatch is *worse* at 74.6%; (b) where coverage does
differ, on Reuters, it tracks the result exactly (GloVe 96.8% cov / 66.35 unseen >
fastText 91.5% / 64.47 > exact 61.28); (c) the fuzzy links that *are* produced on npm are
qualitatively good ("asserts→assert 0.73", "amazon web services→web services 0.91") and
still do not help, because only 572 of 6486 label features lack an exact match and only
348 get any neighbour. Augment mode makes it explicit: +5487 links buys +0.12 all-label
P@1 and costs 4.5 points of unseen P@1. Shortlist recall is pinned at 71.8% in every
configuration, so the npm bottleneck is candidate generation, not label-name embedding.

### label_graph — co-occurrence propagation on top of ZestXML scores (17 min)
`S' = (1-a)S + a·S·G` over a top-k label-label cosine graph, restricted to the existing
candidate support, tuned on an 80/20 split of the **training** points with ZestXML
retrained on the 80% (honest tuning, no test-set search).
**Result: the only method that beats the reference on every all-label metric** — P@5
41.23 (+0.54), nDCG@5 60.77 (+0.64), PSP@5 30.19 (+1.57) — **and contributes exactly
nothing to zero-shot.** Its unseen slice is bit-identical to the baseline (52.11), which I
confirmed to the last decimal.

That identity is not a bug, it is the finding: 0 of the 286 unseen labels have any in- or
out-edge in a graph built from training co-occurrences, by construction. Propagation
cannot reorder them. The agent also caught that the naive update *hurts* — without a
self-loop for isolated labels, `(1-a)S` shrinks unseen labels relative to boosted seen
ones and PSP@5 drops 0.91 below baseline. The submitted variant adds the self-loop. Note
the selection honesty caveat: the highest-validation-P@1 config was the *no*-self-loop
one, and the agent promoted the other on a structural argument after seeing both test
numbers. It reported both verbatim, which is the right thing to do, but "+0.05 P@1" is
inside noise and only the PSP@3/PSP@5 gains are real.

### lowrank — dense low-rank term added to the bilinear scorer (7 min)
`score = xᵀ(W_sparse + UVᵀ)y`, trained jointly on identical shortlists so the low-rank
term is the only difference from a controlled baseline (which reproduced the reference
exactly). **Result: clearly loses — P@1 67.54 vs 73.04.**

Diagnosis is clean: 1.52M dense parameters against 515k sparse ones drive the training
objective from 486k to 74.7k, i.e. memorisation of the 2.47M shortlisted training pairs,
all of which are seen-label pairs. Shrinking the term walks monotonically back toward
baseline (r=32 + 10× L2 → P@1 72.77), so the only non-harmful regime is the one where the
term is switched off.

**⚠ Skeptical note on the 53.16 unseen P@1** — the one number in the sweep that beats the
reference on the zero-shot metric while the method is 5.5 points worse overall. There is a
mechanism (unseen labels' `V` rows are driven by shared `1_<token>` label features, and
re-ranking within the fixed candidate set can improve the unseen-only ordering), so it is
not impossible. But I do not believe it is signal: the agent's own three variants give
unseen 53.16 / 51.48 / 51.96 with no monotone relationship to the term's strength, and
that ±1.7-point spread is the noise floor on a 4786-point slice. Treat it as "unchanged".
Do not cite this as a zero-shot win.

### classical — centroid, kNN, BM25, and a 1:1 hybrid (12 seconds of compute)
The most useful result in the sweep, and the cheapest. **BM25 with zero training reaches
unseen P@1 47.45 on NPM against ZestXML's 52.11 — 91% of the reference's zero-shot
accuracy from naive lexical overlap between the package text and the tag's own name.** On
Reuters, BM25's unseen P@1 is 61.28, which equals the reference to two decimals.

Two things to be careful with here. First, the Reuters coincidence is a coincidence of
counts, not of predictions: 61.28% of 532 points is exactly 326 correct for both systems,
but the two disagree further down the list (ZestXML P@3 26.50 / nDCG@5 74.29 vs BM25
25.81 / 73.24). The agent's phrasing "ZestXML's zero-shot ability on Reuters appears to be
entirely lexical" overstates it by a hair — "almost entirely, at rank 1, on 532 points and
15 labels" is the defensible version. Second, the hybrid's PSP@5 of 36.94 (vs the
reference's 28.62) is real and I verified it, but it is bought with BM25's rare/unseen
mass and it does **not** transfer: on Reuters the same untuned 1:1 hybrid collapses to
P@1 72.84 vs kNN's 81.05, because there BM25's seen-label scores are near-noise and the
fixed weight lets that noise outrank good kNN scores.

Centroid and kNN are pure co-occurrence memorisation and cannot touch unseen labels at all
(see the 2.49 note above). kNN beats centroid on P@1 because npm tags are multi-modal;
centroid beats kNN on PSP because averaging is better behaved for rare labels.

### ova_linear — one-vs-all linear classifiers, no label features (9 min)
Squared hinge, L2 tuned on a held-out 10% of the *training* set. **Result: loses to
ZestXML on all labels (71.18 vs 73.04) and — the interesting part — loses on seen labels
too** (seen P@1 73.09 vs 74.97; seen PSP@1 40.34 vs **47.91**, using the corrected
reference). So it does not establish the "seen-label ceiling" the experiment was designed
to expose: the zero-shot machinery costs nothing in seen-label head accuracy and actively
helps on seen *tail* labels.

The explanation — 95355 positives over 3223 labels is ~30 per label, so per-label problems
are data-starved and ZestXML's shared token representation acts as transfer/regularisation
— is convincing, and the 7.6-point seen-only PSP@1 gap is the direct measurement. Caveats
the agent flagged and I endorse: this is 30 full-batch Adam epochs with a shared L2, not a
converged per-label DiSMEC solve, and the loss was still oscillating. Read seen-only P@1
73.09 vs 74.97 as "roughly a tie on head accuracy". The PSP gap is large enough to
probably survive better optimisation, but that was not verified. Reuters reused the NPM
hyperparameters untuned.

### splade — learned sparse expansion, no pretrained LM (4.5 min)
Learned low-rank expansion on both point and label side over the point-feature vocabulary,
FLOPS-regularised, trained with BCE on sampled pairs, retrieved with a genuine sparse
inverted-index dot product. **Result: loses on every headline metric and collapses on
zero-shot — unseen P@1 23.00 vs 52.11.**

The architecture works as retrieval: sparsity is real and measured (370.9 non-zeros per
point vector out of 17299 before top-k, down from 2128 at step 20 — the regulariser is
doing the work), and seen P@1 72.78 is respectable. What kills it is a structural
overfitting route the agent identified and then demonstrated with a diagnostic run: every
label carries a unique `__label__i__name` feature, so the learned label expansion turns
that one feature into a free per-label vector for any label with positives. At 80 steps
unseen P@1 is 36.79 and seen is 67.71; at 576 steps unseen has fallen to 23.00 while seen
has risen to 72.78. Training monotonically trades zero-shot generalisation for seen-label
memorisation. This is the cleanest evidence in the sweep that **it is the pretrained MLM's
lexical priors, not the sparse-expansion architecture, that makes real SPLADE work
zero-shot** — learning the expansion from 25127 points with ~10.7 non-zeros each gives it
no way to relate tokens that never co-occur.

The agent correctly refused to promote the 80-step smoke run as its headline, since the
only legitimate way to select it would be a pseudo-unseen validation split it did not have
budget to build. Named fixes it did not get to: drop or heavily penalise the per-label id
feature, L2-normalise label vectors to kill the norm/popularity prior.

### dense_probe — retrieval-ceiling probe with pretrained word vectors (4 min)
Not a competitor; a probe of whether dense retrieval could raise the 71.8% candidate
ceiling. **Answer: decisively no, and it fails for two separable reasons that its own
one-hot control pins down.** (1) The word-vector step is itself harmful at K=100: plain
lexical token cosine through the identical pipeline recalls 47.16 micro@100, Numberbatch
drops it to 39.32, fastText to 27.25. npm vocabulary is technical and 21–27% OOV, the
in-vocabulary tokens get blurred toward general English, and tf-idf averaging over
hundreds of features collapses documents toward a corpus centroid. (2) Even the
perfectly-covered lexical control (47.16) is ~25 points below ZestXML's shortlist (71.82),
so the shortlist's advantage is **learned Xf→Yf structure, not name matching**, and no
untrained retriever — dense or sparse — closes that.

The one positive, and the sweep's most actionable lead: dense is *complementary* on unseen
labels. Union of dense-Numberbatch top-50 with lexical top-50 reaches **57.28 unseen micro
recall vs the full lexical top-100's 54.01**, at the same 100-candidate budget, paying
3.2 points of seen recall. As a ranker, pure dense cosine is far behind everything
(P@1 37.71).

Note the anomaly, which I verified rather than smoothed: dense_probe is the only method
whose seen-only P@1 (34.12) is *below* its all-label P@1 (37.71). That is not an error —
dense cosine systematically ranks specific unseen label names above frequent seen ones, so
masking unseen labels away removes correct top-1 predictions. It is a compact restatement
of the complementarity finding. Its Numberbatch-over-fastText choice was made on the test
split (no dev split exists here) and should be discounted; both tables are reported.

---

## 3. Where the remaining gap is

**Solid — I would defend these.**

1. *The npm zero-shot ceiling is candidate generation, not label-name representation.*
   Shortlist recall@100 is 71.8% in every single configuration tried: exact map, fastText
   fallback, fastText augment, Numberbatch either way, low-rank re-ranking, graph
   propagation. Nothing in this sweep moved it. Every re-ranking method is therefore
   fighting over a fixed 71.8% ceiling and a 3223-way competition.
2. *A large majority of ZestXML's zero-shot ability on these datasets is lexical.* BM25
   with literally zero training gets 47.45 / 52.11 of the reference's unseen P@1 on NPM
   and matches it at rank 1 on Reuters. The learned bilinear part buys ~4.7 unseen P@1 on
   NPM and very little on Reuters.
3. *Co-occurrence-based methods are structurally incapable of zero-shot*, and their 2.49 /
   2.07 readings are an evaluator tie-break, not a score. Verified at the matrix level.
4. *ZestXML's shared-token representation is a strong regulariser for rare SEEN labels,
   not just a zero-shot device.* OVA loses 7.6 points of seen-only PSP@1 to it with ~30
   positives per label.
5. *Dense word-vector retrieval is worse than lexical retrieval on npm, at equal K.* The
   one-hot control makes this a controlled comparison rather than an impression.
6. *Label co-occurrence propagation cannot help unseen labels* — 0 of 286 have any edge.
   Proof by construction, and the unseen slice is bit-identical before and after.

**Suggestive — directionally believable, not established.**

7. *Learned sparse expansion without a pretrained LM cannot do zero-shot.* One
   architecture, one seed, one hyperparameter setting, no validation split. The training
   dynamic (unseen 36.79 → 23.00 as seen 67.71 → 72.78) is a strong hint that the failure
   is structural rather than a tuning miss, but the named fixes (drop the per-label id
   feature, normalise label vectors) were never run, so "SPLADE-without-BERT fails at
   zero-shot" is currently "this SPLADE-without-BERT failed at zero-shot".
8. *Adding dense candidates raises the unseen ceiling.* +3.3 unseen micro recall at equal
   budget is measured, but only end-to-end as recall — nobody ran the pipeline on the
   union shortlist, so the downstream P@1/PSP effect is unknown. Note this sits in mild
   tension with `better_vectors`, where injecting semantic links into *scoring* hurt
   unseen precision; the resolution is presumably that extra semantic mass helps recall
   and hurts ranking, but that is a hypothesis, not a result.
9. *Label-graph propagation's +1.57 PSP@5.* Honestly tuned and consistent between
   validation and test, so probably real, but it is one dataset and the P@1 movement is
   noise.
10. *The classical hybrid's PSP@5 36.94 > reference 28.62.* Verified on NPM, but the same
    untuned hybrid loses 8 points of P@1 on Reuters, so it does not generalise as a recipe.
11. *OVA's seen-label near-tie.* An unconverged 30-epoch Adam solve; a proper LIBLINEAR
    per-label solve would plausibly add 1–2 points.

**Could not be tested at all — do not draw conclusions here.**

12. **Any pretrained transformer.** HuggingFace is blocked from this sandbox, so the whole
    class the field currently uses (dense bi-encoders, cross-encoder re-rankers, real
    SPLADE with MLM-initialised expansion, T5/GPT label generation) is untouched. The
    single most important untested hypothesis is whether an MLM's lexical priors close the
    npm unseen gap that static word vectors cannot.
13. **Real fastText with subword backoff.** We only had the 1M-word text table, which is
    subword-*trained* but not subword-*queryable*. The npm vocabulary is compound and
    versioned (`a11y`, `webpack`, `a6cab4d`); hashed character n-grams from the binary
    model are the one remaining door for the coverage hypothesis, and it stayed closed.
14. **GPU-scale training.** Everything here is CPU, 2 threads, and mostly under 10 minutes
    per method. The low-rank and SPLADE runs in particular are compute-starved.
15. **Published XMC benchmarks.** GZ-NPM and GZ-Reuters-90 are self-built. Reuters has 90
    labels, 15 of them unseen, and 532 evaluable unseen points — several "findings" there
    rest on a few hundred decisions. Nothing in this sweep is comparable to published
    LF-AmazonTitles / EURLex / Wikipedia numbers, and no conclusion about ZestXML's
    standing versus the literature can be drawn from it.
16. **Reuters for four of six methods.** splade, dense_probe, lowrank and label_graph
    never ran it. Where a table cell is empty it is unmeasured.
17. **Variance.** Every method is a single seed with no error bars. On the 4786-point
    unseen slice the one-sigma binomial width is roughly ±0.7 points, so every unseen-P@1
    difference under ~1.5 points in this report — including lowrank's "+1.05" and
    label_graph's "+0.05" P@1 — is uninterpretable.

---

## 4. What to run next, in order

1. **Fix the SPLADE label-side leak and re-run.** Drop (or heavily L2-penalise) the
   per-label `__label__i__name` feature so the label side must route through shared
   tokens, L2-normalise label vectors, and early-stop on a held-out set of *pseudo-unseen*
   seen labels. *Payoff:* turns a confounded negative into a clean test of whether the
   sparse-expansion architecture itself can do zero-shot; the diagnostic run says unseen
   P@1 in the 35–40 range is reachable. *Needs:* nothing — CPU, ~10 min, code already
   written.
2. **Attack candidate generation, not ranking.** Build the shortlist as
   lexical-top-50 ∪ dense-top-50 and run the full ZestXML pipeline on it. *Payoff:* the
   only measured route past the 71.8% ceiling that every re-ranker is stuck behind
   (+3.3 unseen recall at equal budget). *Needs:* CPU, the already-downloaded Numberbatch
   table, one pipeline run.
3. **Real fastText binary with subword backoff for the fuzzy direct map.** *Payoff:*
   closes the single open door in the coverage hypothesis — npm's OOV tokens are exactly
   the compound/versioned strings character n-grams are for. Reuters already shows the
   mechanism fires when coverage is there (+5 unseen P@1 over exact). *Needs:* the ~7 GB
   `cc.en.300.bin` download; CPU only.
4. **A converged per-label OVA (LIBLINEAR / DiSMEC-style, per-label C).** *Payoff:*
   settles whether ZestXML's seen-label tie is real or an artifact of an unconverged Adam
   baseline — currently the weakest link in the "zero-shot machinery is free" claim.
   *Needs:* CPU, ~1 hour, liblinear.
5. **Pretrained-transformer label encoder** (MLM-initialised SPLADE, or a bi-encoder over
   label names with ZestXML's sparse scorer as the ranker). *Payoff:* the largest expected
   move on unseen P@1, and the direct test of the "pretrained lexical priors are the
   load-bearing part" hypothesis that runs through three of the six reports. *Needs:*
   **unblocked HuggingFace + GPU** — cannot start today.
6. **Re-run the two most interesting results on a published XMC benchmark** (LF-Amazon-131K
   or EURLex-4.3K zero-shot splits): the BM25-recovers-91%-of-zero-shot finding, and the
   OVA seen-label comparison. *Payoff:* tells us whether these are properties of
   zero-shot XMC or artifacts of two self-built datasets — right now we cannot tell.
   *Needs:* **benchmark data** (download + preprocessing), CPU sufficient.
7. **Cheap hygiene, do it alongside anything above:** delete or clearly label the stale
   `Results/GZ-NPM-torch` matrix so no future agent compares against 71.99 again; run the
   four missing Reuters configurations; put 3 seeds behind every unseen-P@1 claim.
   *Payoff:* removes two live sources of wrong conclusions in this very report.
   *Needs:* CPU, under an hour.

---

# Embedding hybrids (workflow `zestxml-embedding-hybrid`)

Two ways of complementing the lexical zero-shot bridge with embeddings, each implemented in
an isolated worktree, measured against a re-run control, and then handed to an adversarial
verifier that re-evaluated every artifact and tried to refute the claim. Both verifiers
reproduced every reported number to 0.00.

## Diagnostic (why the obvious framing was wrong)

| | GZ-NPM | GZ-Reuters-90 |
| --- | --- | --- |
| label-name tokens present in `Xf` | 89.59% | 82.88% |
| **unseen**-label tokens present in `Xf` | **99.09%** | **100%** |
| mean tokens per label name | 1.20 | 1.23 |
| unseen test positives inside the control shortlist | **54.01%** | 70.93% |
| overall shortlist recall@100 | 71.82% | 95.19% |

Two things fall out. Unseen labels are already almost perfectly covered lexically — the
lower all-label figures come from *seen* labels with rare or junk tokens, where it does not
matter — so "expansion buys coverage" was the wrong theory. And **nearly half of npm's
unseen positives never enter the candidate set**, which caps any hybrid that only rescores
candidates, independently of how good the new channel is.

## 1. Label feature-bag expansion — ships, opt-in

`build_dataset(..., label_expand=glove_expander(...))`. Unseen-label P@1 sweep on
GZ-Reuters-90 (control 61.28):

| k | cosine floor | unseen P@1 | all P@1 | seen P@1 |
| --- | --- | --- | --- | --- |
| 2 | 0.5 | 67.29 | 86.09 | 94.93 |
| **2** | **0.7** | **69.36** | **86.75** | 95.08 |
| 5 | 0.5 | 65.60 | 86.25 | 94.89 |
| 5 | 0.7 | 65.79 | 86.98 | 95.01 |

GZ-NPM at the same k=2 / 0.7: unseen P@1 **52.11 → 44.88**, all-label 73.04 → 73.13.

Verification: reproduces at seeds 0/1/2 with delta +8.08 each time against a 0.38 seed
spread; the two arms differ in `Yf.txt` and `Y_Yf.txt` alone (byte-compared, 11 of 13 files
identical), so the unseen label set, points and splits are shared; `Xf` is fit on training
text only and no test data enters the expansion. Corrections the verifier made to the
implementing agent: the seen-label cost is ≈ −0.1 across seeds, not the exact 0.00 seed 0
shows; and the variant creates 67/532 top-1 ties on the unseen split against the control's
2, so tie-neutral accounting gives +7.56 rather than +8.08 — the evaluator's deterministic
order is the *pessimistic* reading for the variant. The operating point was selected on the
test split over 4 cells; all four beat control (+4.3 / +8.1 / +4.3 / +4.5), so the expected
gain is ≈ **+5.7 to +7.6**, not +8.08.

The predictor of the sign is how widely an added feature is shared, printed by
`build_dataset`: Reuters 1.25 labels per added feature (max 6, `wheat`/`corn`); npm 1.54
(max 19, `example` on 19 labels, `internet` on 13). npm's shortlist recall went *up*
(71.82 → 72.49), so the regression is entirely in scoring, not retrieval.

## 2. Fused dense scoring term — refuted, does not ship

`score = a·bilinear + b·knn + c·cos(enc(doc), enc(label))`, GloVe tf-idf-weighted means,
scored only on shortlist pairs.

| dataset | arm | P@1 | PSP@1 | PSP@5 | unseen P@1 |
| --- | --- | --- | --- | --- | --- |
| GZ-NPM | control | **73.04** | 19.10 | 28.62 | **52.11** |
| GZ-NPM | learned fusion (a=1.00, b=0.00, c=0.00) | 72.21 | 18.33 | 25.19 | **5.27** |
| GZ-NPM | oracle, c tuned on test | 72.85 | 19.01 | 29.95 | 32.87 |
| GZ-Reuters-90 | control | **86.35** | 43.97 | 62.94 | **61.28** |
| GZ-Reuters-90 | learned fusion (0.50/0.10/0.40) | 84.17 | 37.03 | 63.00 | 52.26 |
| GZ-Reuters-90 | oracle, full simplex on test | 87.02 | 50.02 | 71.70 | 59.96 |

Every honest comparison is a loss, and on npm even the test-tuned oracle stays below
control. The controls are unusually tight: all `model/` artifacts are bit-identical to
`Results/Npm2exact/model` and `Results/Reu2exact/model`, and the arms differ by exactly the
fusion.

Why it failed, and what to fix if it is retried:

* **The fitting is structurally broken, not just badly tuned.** The validation points come
  from the set the classifier was trained on, so validation P@1 is saturated (96.86 npm /
  99.61 Reuters), the search reads the bilinear term as near-perfect and zeroes the `knn`
  term — which *is* the zero-shot mechanism, hence unseen 5.27. Worse, a validation
  shortlist built on training data contains **no unseen labels by construction**, so no
  tuning on it can select for unseen performance. A validation split must be held out
  before the classifier is trained, with unseen labels present.
* **The 5.27 is a real ranking collapse, not a masking artifact.** The verifier checked the
  matrix: 33549 non-zero entries in the 286 unseen columns, identical support to the
  control. npm's tie-break floor is 2.49; this is above it and genuinely scored.
* **Most of the one positive row is not the dense term.** Deleting the dense channel and
  re-tuning only the lexical mix on standardised channels
  (`Results/Refute2reu-nodense/score_mat.bin`) recovers 8.11 of the 8.76 PSP@5 gain and
  1.58 of the 6.05 PSP@1 gain — at P@1 86.22, still below control. The dense channel's
  marginal contribution under oracle conditions is +0.80 P@1, and it is unavailable without
  test labels.
* **Per-point standardisation is itself lossy**: the same 0.9/0.1 mix scores 86.35 raw and
  84.47 standardised, because the cross-candidate magnitude of the `knn` term is what
  carries unseen labels.

## Where the remaining headroom is

Not in rescoring. 46% of npm's unseen test positives never reach the candidate set, so the
ceiling for any fusion or expansion that reranks a fixed shortlist is 54% unseen recall.
Candidate generation for unseen labels is the binding constraint.

---

# RQ-KMeans semantic tokens (workflow `zestxml-rq-kmeans-tokens`)

Residual-quantization k-means over GloVe embeddings, emitted as features: a document gets
`rq0_<c> ... rq{L-1}_<c>`, a label gets `1_rq0_<c> ...`. The direct map already strips up to
the first underscore and matches on string equality, so the two link with no model change —
a label and a document share a feature when they land in the same quantization cell rather
than only when they share a literal word. The RQ-KMeans variant of the RQ-VAE semantic-ID
idea (TIGER / OneRec), with plain k-means at each level instead of a learned codebook.

Five agents, two adversarial verifiers, plus four arms I ran myself. Both verifiers
reproduced every reported number and confirmed the codebook is fit on **training documents
only** (`rq_kmeans(trn_emb, ...)`; test documents and label names go through
`transform()`, which only does `pairwise_distances_argmin` against stored centroids).

## GZ-Reuters-90 — every arm measured, control P@1 86.35 / unseen 61.28 / seen 95.04

| arm | P@1 | unseen P@1 | seen P@1 | unseen shortlist recall |
|---|---|---|---|---|
| control (lexical) | **86.35** | 61.28 | 95.04 | 70.93 |
| replace, L=4 K=64 | 72.11 | 24.25 | 81.18 | 80.23 |
| replace, L=4 K=32 (best swept cell) | 71.91 | 12.22 | 80.88 | 78.90 |
| augment w=1.0 | 83.41 | 44.17 | 93.51 | 88.54 |
| augment w=0.5 | 84.27 | 51.32 | 94.48 | 94.02 |
| **augment w=0.3** | 84.83 | **62.97** | 94.56 | **94.68** |
| **augment w=0.3, label side only** | **86.55** | 62.22 | 95.04 | 93.36 |
| concat blocknorm, 10% mass | 86.25 | 60.90 | 94.86 | 81.06 |
| concat blocknorm, 25% | 84.86 | 58.08 | 94.37 | 76.58 |
| concat idf, 25% | 85.59 | 59.77 | 94.63 | 81.23 |
| concat blocknorm, 50% | 84.50 | 46.99 | 94.22 | 86.38 |
| prefix tuples, 10% mass | 86.19 | 60.53 | 94.63 | 73.42 |
| prefix tuples, 30% mass | 84.66 | 59.77 | 94.19 | 73.75 |
| lexical model, **rq shortlist only** | 86.45 | 59.96 | 95.08 | 81.06 |
| lexical model, **union shortlist** | 86.39 | 60.53 | 95.04 | 83.39 |

## What holds

* **Replace is dead.** Reproduced independently twice: unseen P@1 61.28 → 24.25 at L=4 K=64,
  → 12.22 at the best cell of a six-cell geometry sweep. In replace mode the row normalises
  to L equal entries, so there is no weight left to tune. Do not pursue it.
* **Weight is the whole variable.** Unseen P@1 across the augment weights: 44.17 (w=1.0),
  51.32 (0.5), 62.97 (0.3). The first conclusion drawn here — "retrieval yes, scoring no" —
  was an artifact of an untuned weight, not a property of the method.
* **The gain is label-feature sharing, not retrieval.** Two independent results pin this
  down. Putting codes on the *label side only* preserves the control's P@1 and seen P@1
  exactly while keeping the unseen gain, so re-representing documents contributes nothing.
  And feeding the lexical model **more candidates alone buys nothing**: +12.5 points of
  unseen shortlist recall (70.93 → 83.39, union arm) moved unseen P@1 *down* 0.75. The extra
  candidates arrive as distractors — the scorer has no way to pick them out. Channel 2 (a
  shared feature carrying trained weight) is doing the work, not channel 1.
* **At matched mass, idf beats flat weighting** — 85.59/59.77/81.23 against 84.86/58.08/76.58
  at 25%. Downweighting a widely-shared code helps once it is not allowed to hijack the row.

## What does not hold

* **The Reuters precision gain is marginal, and smaller than it first looked.** A verifier
  re-ran three seeds: overall P@1 for the label-only arm is +0.077 mean against a control
  seed spread of 0.066 — **noise**. Unseen P@1 is +0.94 / +1.32 / +0.75, sign-stable but
  ~1 point, inside the control's own 0.38 unseen-P@1 spread as a single reading, and `w=0.3`
  was chosen on test. Suggestive, not established.
* **The robust gain is PSP@5, not P@1**: +1.74 / +2.11 / +2.00 across seeds, ~30× the
  control's PSP@5 spread. Tail labels, which is where propensity scoring looks.
* **Nothing transfers to GZ-NPM.** Unseen shortlist recall 54.01 → 54.99 (+1.0), overall
  71.82 → 71.90, unseen P@1 52.11 → 51.46 (−0.65). Every delta is a wash. The geometry sweep
  was worse than a wash: its best Reuters cell took npm's overall recall from 71.82 to 22.88.

## Why, mechanically — two wrong answers, then the one that survives

**Wrong answer 1: "codes encode theme, not identity."** This was recorded here on the
strength of one example — `cotton = [41,60,2,48]` and `corn = [41,60,35,48]` sharing 3 of 4
levels. It does not survive re-fitting; sklearn's KMeans is not bit-reproducible in this
environment, and in a re-run the two share only level 0. `benchmarks/rq_probe.py` says close
to the opposite: the 87 coded Reuters labels occupy 18 distinct cells at level 0 but **80–85
distinct full 4-tuples**, and the few collisions are ones a semantic space *ought* to make —
gold/silver, gas/nat gas, cocoa/coffee, crude/veg oil. Coarse cells are coherent: one holds
the agricultural commodities, one the oils and metals, one the macro indicators. The
quantizer was never the problem.

**Wrong answer 2: "the identity is in the conjunction, so emit prefix tuples."** That
follows from the probe, and it is wrong too — measured, not argued. `--tuples` takes Reuters
from 256 rq features to 20544 and gives, at 10% / 30% mass: P@1 86.19 / 84.66, unseen P@1
60.53 / 59.77, unseen recall 73.42 / 73.75. Worse than the marginals it was meant to fix,
and more mass makes it worse. The recall column says why: a near-unique tuple identifies a
label beautifully *when it matches exactly* and almost never matches exactly, so it fires
far less often than a marginal (+2.5 recall against +10.1). Marginals are too coarse to
rank; tuples are too sparse to fire. Neither converts.

**What survives.** The direct map already links `1_corn` to `corn` by exact string equality,
and **99.09% of npm's and 100% of Reuters' unseen-label tokens are already present in `Xf`**
(measured in the diagnostic phase of the hybrid workflow). So for essentially every unseen
label, an exact lexical link already fires. Semantic codes can only add a *fuzzier* version
of a link that is already there in its sharpest form — and the earlier direct-map result
found the same thing independently: fuzzy matching helped only where vector coverage was
poor and hurt where it was good. There is no gap here for semantics to fill. That also
explains the one result that looked like a contradiction: handing the unchanged lexical
model 12.5 points more unseen recall moved unseen P@1 *down*, because the labels it newly
retrieves are ones the exact channel had already considered and correctly declined.

This is the explanation consistent with every arm measured, but it is an explanation, not a
measurement. The test that would settle it: restrict to labels whose tokens are *absent*
from `Xf` and see whether rq codes help there. On these two datasets that subpopulation is
almost empty, which is precisely the point.

## Transformer encoders — the limitation was not the encoder

Everything above used GloVe means, which had a real defect: documents averaged over 200
tokens, label names over one or two, so the two sides sat in different regions and label
names were then quantized against *document*-cluster centroids. The obvious suspicion was
that this, rather than the idea, produced the negative result. Stage 7 of
`benchmarks/colab/ZestXML_benchmarks.ipynb` tests it on a GPU box with one encoder for both
sides and no out-of-vocabulary drops. GZ-Reuters-90:

| arm | P@1 | PSP@5 | unseen P@1 | seen P@1 | unseen recall |
|---|---|---|---|---|---|
| control (lexical) | **86.35** | **62.94** | 61.28 | 95.04 | 70.76 |
| all-MiniLM-L6-v2 @ 10% mass | 86.05 | 62.38 | 59.59 | 94.74 | 77.24 |
| all-MiniLM-L6-v2 @ 25% | 85.66 | 58.40 | 55.64 | 94.78 | 75.58 |
| clip-ViT-B-32 @ 10% | 85.92 | 61.04 | **61.84** | **95.12** | 77.41 |
| clip-ViT-B-32 @ 25% | 85.69 | 56.44 | 48.68 | 95.08 | 78.74 |

The bar, set before the run: an unseen P@1 gain larger than ~1 point. Nothing clears it —
the best is CLIP at 10% mass, **+0.56**, a single-seed reading barely outside the 0.38 seed
spread. **MiniLM, a strictly stronger text encoder, does worse than GloVe** at the same mass
(−1.69 against −0.38), which refutes the asymmetry hypothesis outright. The same monotone
cliff in mass reappears (10% → 25% costs 4–13 points of unseen P@1), and unseen recall rises
again (70.8 → 77–79) without converting — the third independent confirmation, after rq
shortlists and MiniLM dense retrieval.

The cost is visible in the run logs: the sparsity pattern grows from 284,540 to 512,753
non-zeros. Double the parameters, no gain.

One result worth keeping for its own sake: **CLIP beats MiniLM on unseen labels despite
being much the weaker text encoder here** — its 77-token window truncates these documents
hard. A plausible reason, untested, is that CLIP's text tower is trained on short captions
and so embeds one- and two-word label names in a space that suits them, while a MiniLM
embedding of a 200-token news article encodes register more than topic. If a semantic
channel is ever wanted on the label side, that is the lead to follow.

## Unaddressed

The npm null is diluted by 697 labels that received no code at all, and was never split
coded-vs-uncoded — the one measurement that could still rescue the npm result. The weight
sweep is three test-read points. sklearn's KMeans is not bit-reproducible here, so the
w=1.0 arm used a slightly different codebook from the others.


---

# Three changes to *how* semantic IDs are used

Earlier arms all used codes as peer features and tuned their row mass. These change the use
rather than the amount. All numbers below measured by me on CPU with GloVe-100d.

## Multiple codes per label, and retraining on semantic candidates

`--label_codes m` assigns the *m* nearest centroids per level instead of one. Each code
stays as frequent as before -- so it still fires, which is where prefix tuples failed --
while the set becomes specific. `rq_retrain.py` then closes the hole in the earlier
retrieval test, which had reused a model trained on lexically-retrieved negatives to rank
semantically-retrieved ones; it regenerates **both** shortlists and retrains from scratch.

| GZ-Reuters-90 | P@1 | PSP@5 | unseen P@1 | seen P@1 | unseen recall |
|---|---|---|---|---|---|
| lexical control | 86.35 | 62.94 | **61.28** | **95.08** | 70.76 |
| semantic features (MiniLM, m=3) | 86.05 | 63.00 | **61.28** | 94.86 | 84.05 |
| lexical + sem candidates | 86.52 | 64.59 | 60.15 | 94.97 | 84.05 |
| lexical + union candidates | **86.58** | **64.78** | 61.09 | 95.04 | **85.88** |

| GZ-NPM | P@1 | PSP@5 | unseen P@1 | seen P@1 | unseen recall |
|---|---|---|---|---|---|
| lexical control | 73.04 | 28.64 | **52.11** | 74.97 | 54.01 |
| semantic features (GloVe, m=3) | **73.40** | 28.40 | 51.11 | **75.31** | 54.40 |
| lexical + sem candidates | 73.04 | **28.74** | 51.98 | 74.95 | 54.40 |
| lexical + union candidates | 73.04 | 28.64 | 52.09 | 74.97 | **54.90** |

**Multiple codes per label removed the penalty.** On Reuters the semantic-feature arm hits
**exactly** the control's unseen P@1 where every earlier semantic-feature arm lost (GloVe
m=1: 59.59; replace: 24.25). Labels go from ~80 distinct code-tuples to 87 of 87 distinct
code-sets, and 2470 of 2526 on npm even though a level-0 code there lands on 40 labels on
average (max 467). Identity is recoverable; it just does not convert.

**Retraining removed the loss but created no gain.** Unseen P@1 returns from 60.53 to the
control's 61.28, so the earlier -0.75 was a train/test artifact -- but 15 points of extra
unseen recall still produce no unseen P@1. With the mismatch closed, "retrieval is not the
binding constraint" is established rather than suspected.

**npm is a wash on every axis, including retrieval.** Unseen recall moves +0.89 against
Reuters' +15.1. Two measured causes: 697 of 3223 labels receive no code at all under GloVe's
vocabulary, and 64-way cells are far too coarse for 3223 fine-grained labels.

## Semantic pruning of the mined pattern

`-prune_vectors` / `-prune_min_sim` drop mined pairs whose feature names are unrelated. On
GZ-Reuters-90, keeping 78% (min_sim 0.15) and 66% (0.30) of the pattern:

| min_sim | P@1 | PSP@5 | unseen P@1 | seen P@1 |
|---|---|---|---|---|
| 0.00 (control) | **86.35** | **62.94** | **61.28** | 95.04 |
| 0.15 | 86.25 | 61.82 | 60.15 | **95.27** |
| 0.30 | 86.25 | 60.47 | 59.77 | **95.27** |

On Reuters it trades unseen accuracy for seen accuracy and does not pay for itself.

**On GZ-NPM it is the largest overall gain measured anywhere in this repo**, and it inverts
the Reuters conclusion. Keeping 73% of the pattern (min_sim 0.30, 174649 of 470401 entries
judgeable):

| min_sim | P@1 | P@5 | PSP@5 | unseen P@1 | seen P@1 |
|---|---|---|---|---|---|
| 0.00 (control) | 73.02 | 40.71 | 28.67 | **52.07** | 74.95 |
| 0.15 | 73.88 | 41.03 | 28.82 | **52.19** | 75.84 |
| **0.30** | **74.57** | **41.19** | 28.87 | 52.13 | **76.56** |

Measured on a GPU box and reproduced independently here on CPU to within 0.02 (73.05 ->
74.59, seen 74.98 -> 76.58). **+1.55 P@1 and +1.61 seen P@1**, with the unseen split
unchanged -- so this makes the model better, not more zero-shot. Note that 0.30 beats 0.15,
so the sweep has not found its optimum.

Why npm and not Reuters: Reuters mines 284K pattern entries over 90 labels, npm 470K over
3223. `bs_count` keeps the top-k label features per point feature by co-occurrence alone,
and on the larger and sparser label space many of those slots go to pairs that co-occur
without being related. Removing them frees budget that was being spent on noise. The
conclusion first recorded here -- "co-occurrence was already choosing the right pairs" --
was drawn from Reuters alone and is wrong on any label space large enough for the pattern
budget to bind.

## A measurement error worth recording

The first npm run of this section reported control P@1 52.15 against the true 73.04. The
model directory was shared with a previous Reuters run and still held its `seen_labels.txt`;
those 75 ids are all valid indices into npm's 3223 labels, so nothing complained and 2862
labels with real training data were treated as unseen. Shortlist recall matched the
reference *exactly* -- only scoring was wrong -- which is what made it survive a first
glance. The pipeline now rejects a seen-labels cache that disagrees with `trn_X_Y`.


## Label expansion: GloVe beats modern encoders

Same mechanism, same k=2 and cosine floor 0.7, three encoders, GZ-Reuters-90:

| expander | P@1 | PSP@5 | unseen P@1 | seen P@1 |
|---|---|---|---|---|
| **GloVe-100d** | **86.72** | **72.47** | **69.74** | 95.04 |
| all-MiniLM-L6-v2 | 86.22 | 63.51 | 63.35 | **95.12** |
| control (no expansion) | 86.35 | 62.94 | 61.28 | 95.04 |
| clip-ViT-B-32 | 85.62 | 62.65 | 58.46 | 94.93 |

A 2014 word-vector table beats a modern sentence encoder by 6.4 points of unseen P@1, and
CLIP is worse than no expansion at all. The likely cause is that the cosine floor does not
transfer: transformer embedding spaces are anisotropic, so 0.7 is selective for GloVe and
permissive for MiniLM, admitting more and looser neighbours. That is a *threshold* result
rather than an encoder result, and the sweep that would separate them was lost to a
directory-naming collision (both floors wrote to the same tag) -- fixed, not yet re-run.


---

# EMMETT / IRENE: synthesizing the missing classifier

[Yadav et al., KDD '24](https://dl.acm.org/doi/10.1145/3637528.3672046), from the ZestXML
authors, re-implemented compactly in `benchmarks/irene.py`. It inverts every other
experiment here: instead of describing an unseen label better, it builds the classifier the
label does not have out of the classifiers of labels it resembles, through one attention
layer trained on seen labels leave-one-out.

GZ-NPM, all-MiniLM-L6-v2 embeddings, k=16 neighbours, measured on a GPU box:

| arm | P@1 | PSP@5 | unseen P@1 | seen P@1 |
|---|---|---|---|---|
| dual encoder (cosine) | 42.79 | 28.77 | 42.19 | 39.56 |
| one-vs-all | 56.35 | 17.10 | **2.49** (floor) | 57.86 |
| ova, dual encoder on unseen | 52.54 | 37.18 | **42.19** | 57.86 |
| + mean synthesis | 56.35 | 17.10 | 25.60 | 57.86 |
| + IRENE generator | 56.59 | 23.70 | 27.96 | 57.86 |
| dual + mean synthesis | 63.18 | 24.63 | 41.39 | **64.40** |
| **dual + IRENE generator** | **64.17** | **37.72** | 41.25 | 63.83 |
| *ZestXML, for reference* | *73.04* | *28.64* | *52.11* | *74.97* |

**The mechanism works.** One-vs-all sits at the evaluator's tie-break floor on unseen labels
because it has no classifier for them at all; synthesis takes that to 25.60 by averaging
neighbours and 27.96 through the generator. On GZ-Reuters-90 the generator *lost* to the
average, with only 75 seen labels to train on; npm's 2937 is enough.

**The encoder term is not optional.** Scored alone the synthesized classifier looks like a
failure -- 27.96 against the plain encoder's 42.19. Combined, as the method is deployed, the
same classifiers give P@1 64.17, eight points above one-vs-all and twenty-one above the
encoder. The first reading here was a scoring error, not a result.

**The learned generator earns its keep, on the tail.** dual+IRENE against dual+mean is
PSP@5 37.72 against 24.63, +13.1 -- the largest gain from any learned component measured in
this repo. On P@1 the two are within a point.

**But synthesis does not beat the trivial baseline on unseen labels.** Keeping the
classifiers where they exist and falling back to the encoder where they do not -- no
synthesis, no tuning -- reaches 42.19, above dual+IRENE's 41.25. All of the unseen accuracy
here comes from the encoder. What the synthesized classifiers buy is head and tail accuracy,
not zero-shot accuracy.

**And ZestXML still wins at this scale**, by 8.9 P@1 and 10.9 unseen P@1 -- while *losing*
PSP@5 by 9.1. Consistent with the Renee result: encoder-based extreme classification is
being evaluated two orders of magnitude below the label counts these methods are designed
for (npm has 3223; the paper uses 270K-1.3M).

Caveats on this implementation: the encoder is frozen, one-vs-all trains for 15 epochs in
embedding space rather than end-to-end, and the generator is a single attention layer
trained for 60 steps. It tests the *idea*, not the paper's system, and the absolute
comparison against ZestXML should be read with that in mind.


# AmazonCat-13K: the first standard-benchmark run

Everything above is measured on GZ-NPM and GZ-Reuters-90, two datasets built here. No
published paper reports either, so no number above can be checked against anyone else's.
This section exists to close that: AmazonCat-13K is a repository dataset that XMC papers
do report, converted with `benchmarks/datasets/make_xmc.py --unseen_frac 0` (every label
seen, which is the setting published P@k is measured in).

1.19M train / 306K test / 13,330 labels.

| run | pattern nnz | trn recall | tst recall | P@1 | P@3 | P@5 | nDCG@5 | PSP@1 | PSP@5 |
|---|---|---|---|---|---|---|---|---|---|
| title only | 11,192,094 | 87.55 | 82.05 | 76.53 | — | — | — | — | — |
| title + `--content` | 270,402,032 | 98.00 | 92.46 | **93.41** | 79.60 | 64.29 | 86.01 | 54.02 | 72.15 |

Training on the `--content` run: 2749 s over 118.6M (point, candidate) pairs, 98.17%
accuracy, 83.20% on positives. Pattern mining 173 s.

**The title-only run was mis-specified, not weak.** AmazonCat-13K's published numbers are
measured on the product description, not the title — the title-only variants of these
datasets are separate datasets with `-Titles-` in the name. Feeding the description grows
the mined pattern 24x and lifts test shortlist recall 82.05 -> 92.46, and P@1 by +16.9.
Most of what looked like a modelling deficit was a candidate-generation ceiling created by
throwing away the text the benchmark is defined on.

**The comparison band is still unverified.** The notebook prints `XML-CNN ~75 ..
AttentionXML ~95 .. XR-Transformer ~96` for this dataset, and those figures came from a
web-search summary, not from a paper I read — every source that would settle it
(manikvarma.org, the ACM DL, arXiv HTML) is blocked from this sandbox. 93.41 lands where a
sparse linear model plausibly should against transformer methods, above the CNN era and a
point or two below the strongest, but "plausible" is not "checked". Treat this row as an
internal measurement until someone puts a real table next to it.

**What it does establish, independent of the band:** the port trains and predicts at
1.19M x 13K without changes, and its accuracy on a dataset it was not tuned on is in the
same regime as published work rather than an order off. That was the open question behind
"I don't buy numbers", and it is now answered for scale and sanity, not for ranking.

Not yet run on this dataset: the `--unseen_frac 0.1` zero-shot split, which is the setting
every other experiment in this file is about.


# ZestXML vs SASRec, as an XMC task (ml-1m)

`benchmarks/colab/zestxml_vs_sasrec_standalone.py`. A point is a user; its labels are the
set of items touched in the next 5 steps. 10% of the catalogue is removed from training
entirely — every occurrence deleted from every training history, surviving only inside test
windows, so a cold item has exactly zero training interactions. 6040 users, 3706 items, 371
cold, 5.00 labels per point, 3690 of 30200 positives on cold items. Shortlist recall:
78.76% train, **48.88% test**.

Head/tail/cold rows mask every label outside the group, the way the reference splits seen
from unseen.

| model | group | P@1 | P@5 | nDCG@5 | PSP@5 |
|---|---|---|---|---|---|
| zestxml | all | 8.16 | 6.15 | 6.57 | 3.07 |
| **sasrec** | all | **12.88** | **10.36** | **10.89** | **5.31** |
| sasrec+content | all | 12.88 | 10.25 | 10.83 | 5.28 |
| zestxml | head | 8.44 | 6.34 | 9.58 | 10.37 |
| sasrec | head | **14.25** | **10.91** | **16.72** | **18.15** |
| zestxml | tail | 5.09 | 3.18 | 5.74 | 6.54 |
| sasrec+content | tail | **8.15** | 6.03 | **11.26** | **12.33** |
| zestxml | **cold** | 1.07 | 1.07 | 2.66 | 4.17 |
| sasrec | **cold** | 0.97 | 0.59 | 1.56 | 2.30 |
| **sasrec+content** | **cold** | **1.91** | **1.93** | **4.88** | **7.53** |

**SASRec wins everywhere, including cold — which reverses the earlier reading here.** An
earlier version of this comparison used leave-one-out next-item with Recall@k over the full
catalogue and reported plain SASRec at *exactly* 0.00 on cold items against ZestXML's 1.34,
concluding that the text channel owned the cold column. Under the XMC framing that is
wrong: the content-fed SASRec reaches P@1 1.91 and PSP@5 7.53 against ZestXML's 1.07 and
4.17 — roughly 1.8x on both.

**Why the two framings disagree.** The group rows mask every label outside the group, so
the cold row asks "rank the 371 cold items", not "surface a cold item against 3706". The
second question has a floor of zero for a model whose cold embeddings were never touched;
the first does not. Both are legitimate; they are not the same measurement, and the earlier
conclusion was an artifact of only ever asking the second.

**One number in that table is not yet interpretable.** Plain SASRec scores 0.97 P@1 on cold
items despite having no trained parameters for them. With 371 cold labels and 1.28 cold
positives per qualifying point, uniform chance is about 0.35% — so 0.97 is roughly 2.8x
chance, at about 5 sigma over 2884 points. The likely mechanism is not transfer but
degenerate ordering: a fixed ranking that happens to put a test-frequent cold item first
scores well without knowing anything. `random` and `popularity` control rows have been
added to the runner for exactly this reason; until they are in the table, no cold number
above should be read as evidence of transfer.

**What this suggests about the mechanism.** A learned projection from tf-idf into the
sequence encoder's embedding space does the same job as ZestXML's shared label features,
and does it better here, because it is trained jointly with the encoder rather than bolted
onto a bag-of-words scorer. That is the same lesson as the IRENE result recorded above --
the encoder term is not optional -- arriving from a different direction.

**Caveat that bounds all of it:** test shortlist recall is 48.88%, so ZestXML could not
retrieve more than half the positives no matter how well it ranked. `shorty_k` needs
raising until that plateaus before the gap is attributed to the model.


## Why ml-1m was close to a worst case, measured

The `profile()` row now printed before every run, on ml-1m:

    items 3706, interactions/item mean 255.6 median 112, 13.0% under 10
    title tokens: 3382 distinct, a token is shared by 4.3 items on average;
      mean over items of that figure is 448.8
    labels/point 5.00

Three numbers decide whether a text channel can beat an id embedding, and ml-1m sets all
three against it. A median of 112 interactions per item makes a per-item embedding trivially
learnable. Only 13% of items sit under 10 interactions, so there is almost no tail for
shared features to rescue. And **the average item's title tokens are shared by 449 other
items** -- `Toy Story (1995) Animation Children's Comedy` is three genre words and a name --
so the label feature bag cannot identify anything, and all the discriminative power sits in
the per-label identity feature, which is exactly what a cold item does not have.

Thinning interactions to 20% moves density hard (47.2 per item, 30.8% under 10) and barely
moves discriminativeness (421.2). So on this dataset the binding problem is the *text*, not
the sparsity, and no amount of thinning will make it a fair test. `keep_frac` is in the
runner to check that claim rather than assert it.

## The cold column, with its floor

Uniform random scores, same masking, same metric:

| group | random P@1 |
|---|---|
| all | 0.18 |
| head | 0.63 |
| tail | 0.13 |
| **cold** | **0.31** |

Against that floor the cold row of the table above reads very differently:

| model | cold P@1 | x chance |
|---|---|---|
| sasrec | 0.97 | 3.1x |
| zestxml | 1.07 | 3.4x |
| **sasrec+content** | **1.91** | **6.1x** |

**Neither ZestXML nor plain SASRec meaningfully solves cold-start on ml-1m** -- both sit
around 3x a chance baseline on a 371-label problem, which is a weak signal, and the earlier
reading of ZestXML's 1.07 as a win over SASRec's 0.97 was comparing two numbers that are
both close to the floor. Only the content-projected SASRec is clearly above it. This is the
control that should have been in the first table.


## Amazon Video_Games: the cold column, on a dataset where the text carries identity

Same harness, 8/20-core, `n_cats=0`, horizon 3, cold_frac 0.05. 13694 users, 2885 items,
4000 evaluated, 3.00 labels per point, 692 of 12000 positives cold. Profile:

    items 2885, interactions/item mean 49.1 median 30, 3.4% under 10
    title tokens: a token is shared by 6.2 items on average;
      mean over items of that figure is 151.4

Compare that last figure to ml-1m's 448.8 and to the first, broken Amazon run's 3812.3. It
is the single number that decides whether a label feature bag can identify anything.

| model | all P@1 | head P@1 | tail P@1 | **cold P@1** | **cold PSP@5** |
|---|---|---|---|---|---|
| random | 0.10 | 0.32 | 0.00 | 0.46 | 2.60 |
| popularity | 1.70 | 2.20 | 0.09 | 0.31 | 3.03 |
| zestxml | 3.05 | 4.31 | 2.04 | **4.92** | **14.31** |
| **sasrec** | **4.10** | **5.67** | 2.57 | 0.46 | 3.18 |
| sasrec+content | 3.88 | 5.57 | **2.78** | 0.77 | 4.77 |

**The trade is now demonstrated rather than asserted.** SASRec wins every column with
training signal behind it -- all, head, tail. ZestXML wins cold by 6.4x on P@1 and 3.0x on
PSP@5 against the strongest baseline, and by 10.7x on P@1 against chance. Plain SASRec sits
exactly at the random floor there, which is what a model with untrained embeddings for
those items should do.

**Content projection is not a substitute for scoring through the label's own words.**
`sasrec+content` gets the same tf-idf text ZestXML gets, through a learned linear map. It
buys the *tail* (2.78 against plain SASRec's 2.57, the best tail number in the table) and
almost nothing on cold (0.77 against 4.92). A projection trained on warm items transfers to
rare ones; it does not manufacture a representation for an item the encoder has never seen.

**Why this contradicts the ml-1m section above.** There, `sasrec+content` beat ZestXML on
cold and both sat near the floor. The difference is entirely the text: ml-1m's average item
shares its title tokens with 449 others, so the feature bag carries no identity and only
the per-label identity feature -- exactly what a cold item lacks -- discriminates. The
ml-1m conclusion was a statement about movie genres, not about the method.

**Caveat bounding ZestXML's warm columns:** test shortlist recall is 48.61%, so head and
tail are retrieval-capped rather than ranking-limited. The `SHORTY_K` sweep is in the
notebook; the cold column will not move much, since 144 labels are easy to shortlist.
