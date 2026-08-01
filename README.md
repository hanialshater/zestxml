<img src="Resources/KDD_Logo.jpg" height="100" align="right"/>

# ZestXML in PyTorch — Generalized Zero-Shot Extreme Multi-Label Learning

A PyTorch implementation of [Generalized Zero-Shot Extreme Multi-label Learning](http://manikvarma.org/pubs/gupta21.pdf) ([KDD 2021](https://www.kdd.org/kdd2021/)).
> [Nilesh Gupta](https://nilesh2797.github.io/), [Sakina Bohra](https://www.linkedin.com/in/sakina-bohra-aa46b174/?originalSubdomain=in), [Yashoteja Prabhu](https://vervenumen.github.io/), Saurabh Purohit, [Manik Varma](http://manikvarma.org/)

Extreme Multi-label Learning ([`XML`](http://manikvarma.org/downloads/XC/XMLRepository.html))
assigns the most relevant subset of an enormous label set to a data point. Generalized
Zero-shot XML (`GZXML`) is the version where **the label universe contains both seen and
unseen labels**, and the unseen ones have no training point at all.

ZestXML reaches them because **a label is a bag of text features, not an id**. Its
parameters live in the label-feature vocabulary, so a label named "trail running" is
scoreable the moment it exists, without a single example.

## Install and run

```shell
pip install -r requirements.txt
```

```python
from zestxml import build_dataset, ZestXML

build_dataset(
    "GZXML-Datasets/MyData",
    trn_texts=["hiking boots waterproof shell", ...],   # one string per training point
    trn_labels=[["hiking", "outdoors"], ...],           # label NAMES per point
    tst_texts=[...], tst_labels=[...],
)

ZestXML("GZXML-Datasets/MyData", "Results/MyData").run()   # fit + predict + evaluate
```

`run()` prints P@k, nDCG@k and PSP@k over all labels and, separately, over the labels that
had no training point — the split this method exists for. `examples/quickstart.py` is the
same thing as a runnable file, from raw strings to metrics.

Every hyper-parameter is a keyword argument under the name it has on the command line, and
an unknown one raises rather than being ignored:

```python
model = ZestXML("GZXML-Datasets/MyData", "Results/MyData",
                shortyK=100, bs_count=40, bs_direct_wt=0.8,
                bilinear_classifier_cost=5, device="cuda")
model.fit()                      # stage 1 mines the sparsity pattern, stage 2 trains
scores = model.predict()         # sparse CSR, also written to Results/MyData/score_mat.bin
metrics = model.evaluate()       # {"all labels": {...}, "unseen only": {...}, "seen only": {...}}
```

The command line front end takes the same names and is what the shell scripts call:

```shell
python run_torch.py -trn_X_Xf ... -Y_Yf ... -type all -device cuda
./run_torch.sh GZ-Eurlex-4.3K all
python tools/eval_xc.py Results/GZ-Eurlex-4.3K/score_mat.bin GZXML-Datasets/GZ-Eurlex-4.3K
```

## The layout

| path | what |
| --- | --- |
| `zestxml/` | the implementation — nothing else is needed to train or predict |
| `zestxml/dataset.py` | `build_dataset`: raw texts and label names → a dataset directory |
| `zestxml/api.py` | `ZestXML`: fit / predict / evaluate |
| `zestxml/pattern.py` | stage 1, mining the sparsity pattern of `W` |
| `zestxml/model.py` | the sparse bilinear scorer, shortlisting, the training loop |
| `zestxml/pipeline.py` | the three stages end to end |
| `zestxml/csr.py` | CSR sparse primitives on torch tensors |
| `zestxml/eval.py` | P@k, nDCG@k, PSP@k, split by seen / unseen |
| `zestxml/embed.py` | word vectors and character n-grams for fuzzy label matching |
| `examples/`, `tests/` | a worked example and the test suite |
| `benchmarks/` | everything comparative — baselines, dataset builders, Colab notebooks |

## The model

`score(x, y) = x^T W y + b`, where `W` (*num_Xf × num_Yf*) is confined to a sparsity
pattern mined from feature co-occurrence, so the free parameters are one vector over that
pattern's non-zeros. The final score fuses that bilinear term with an untrained `knn` term
that measures how well a label's own tokens match the point's:
`score = alpha * bilinear + (1 - alpha) * knn`, with `alpha = -score_alpha`.

An unseen label is reachable through three channels: the direct map linking `1_hiking` to
the document feature `hiking`, the learned weights of any label feature it *shares* with a
seen label, and the untrained `knn` term.

This implementation differs from the C++ reference it was ported from in three deliberate
ways. The port was validated against that reference before it was removed — pattern mined
entry for entry, `.bin` models loaded in both directions, scores matched to 1e-5 — and the
`.bin` format is still byte-compatible, so score matrices move between the two.

* **Training.** The reference flattens every shortlisted pair into an `|pairs| × |w|`
  "linear form" matrix and runs liblinear's dual coordinate descent. Here the same primal
  objective, `0.5*||w||^2 + sum_i C_i * loss(y_i * w.phi_i)` with squared hinge
  (`bilinear_classifier_kind=0`) or logistic (`1`) loss, is minimised with Adam. The linear
  form is never materialised: scores contract the three sparse operands directly through a
  sorted-key join, so peak memory depends on the batch rather than on the label or feature
  count. `bilinear_classifier_maxitr` is the epoch count, `batch_size` the points per
  gradient step, `lr` the step size (linearly decayed to 0). Each epoch prints the
  objective, directly comparable to the reference: on synthetic data the defaults reach
  43270 against dual coordinate descent's 43303, with the same train accuracy to within
  half a point. Raise `lr` or `bilinear_classifier_maxitr` if it is still falling at the end.
* **Shortlisting** is exact. The reference walks an inverted index with a shrinking
  threshold to approximate the top-`shortyK` labels; here `X_Xf @ pattern @ Y_Yf^T` is
  computed in chunks and top-k'd exactly, which can only improve shortlist recall.
* **Batch sizing** is by cost, not row count: `max_elems` caps the non-zeros expanded per
  batch and `dense_elems` the size of any dense working block. Lower them if you run out of
  memory, raise them for throughput.

Torch-only parameters: `device` (`auto|cpu|cuda[:n]`), `lr`, `batch_size`, `seed`,
`max_elems`, `dense_elems`, `float64`, and `num_thread=0` to let torch pick.

One consistency fix: with `bilinear_normalize=1` the reference divides the linear-form
features by their norm while training but not while predicting; this applies the same
normalisation in both. Every shipped run script sets it to `0`, where the two agree.

Parameters kept for command line compatibility but inert: `F` (only used by the
approximate shortlist); `propensity_A`, `propensity_B` and `binary_relevance`, which are
inert in the reference too because `create_Xf_Yf_map` binarises `trn_X_Y` after
`ips_weight` has written to it. `bilinear_add_bias=1` is rejected rather than silently
ignored; no run script uses it.

### Two reference behaviours this does not reproduce

Both were found by diffing against the C++ on synthetic data, and both are bugs rather
than modelling choices, so this code computes the intended quantity instead.

* **Dropped score contributions.** `prod_helper` marks an accumulator slot as "seen" with
  `if (sum[id] == 0) indices.push_back(id)`. A label feature whose running sum is still
  exactly zero when a second point feature touches it is pushed twice; `prod` then writes
  the real value for the first copy and sets `sum[id] = 0`, leaving the duplicate holding
  zero, and `sparse_prod` assigns `mask[yf]` entry by entry so the trailing zero wins and
  that label feature contributes nothing. Dual coordinate descent leaves many weights at
  exactly `0`, which is what makes a zero prefix reachable. On the synthetic run it
  silently changed 2 of 24000 test scores; every other pair matched to 1e-5.
* **Ties at the top-k cut.** `prod_for_jaccard` truncates each row with `std::sort`, which
  is unstable, so when several feature pairs share the k-th best score the pattern kept is
  arbitrary.

## Building a dataset

`build_dataset` writes every file `run_torch.py`, `tools/eval_xc.py` and encoder baselines
such as Renee read. It exists because the conventions are not obvious and are easy to get
wrong:

* Each label becomes a bag of features — `1_<token>` per token of its name, plus one unique
  `__label__<i>__<name>`. The `1_` prefix is load-bearing: the direct map strips everything
  up to the first underscore and looks the rest up in the point vocabulary, which is the
  only way a label with no training example is reachable. Descriptive names matter —
  "trail running" beats "cluster_47".
* **Every** token is emitted, in the document vocabulary or not. Emitting only matched
  tokens makes every label trivially matchable and deletes the population zero-shot is about.
* Text files are verified to stay row-aligned with the matrices after being written and
  read back, because a stray `\r` inside a description silently becomes an extra row — and
  Renee, for one, maps line N to row N without ever checking.
* tf-idf is fit on training text only.

For a benchmark-style split, `select_unseen_labels` picks labels to strip from `trn_X_Y`
while keeping their features and their test positives, which is what makes them zero-shot
rather than absent. `benchmarks/datasets/make_reuters.py` and `make_npm.py` are worked
examples; the API reproduces both datasets byte for byte.

## Fuzzy label-to-document matching (off by default)

The direct map is the only bridge an unseen label has to the document vocabulary, and in
the reference it is *string equality*: `acq` never reaches `acquisition`. `direct_map`
replaces equality with a similarity search over feature names:

```shell
-direct_map charngram                        # tf-idf over character n-grams, no downloads
-direct_map vectors -direct_vectors w2v.txt  # cosine in a word2vec/GloVe/fastText space
-direct_topk 3 -direct_min_sim 0.5 -direct_fallback 1
```

Links are weighted by `bs_direct_wt * cosine`, so a loose match counts for less than an
exact one, and `direct_fallback=1` (the default) only searches for label features with no
exact match, leaving correct matches undiluted.

**Results are mixed, and the pattern is instructive.** Unseen-label P@1:

| direct map | Reuters (97% vector coverage) | npm (77% coverage) |
| --- | --- | --- |
| `exact` (reference) | 61.28 | **52.11** |
| `charngram`, fallback | 61.28 | 51.76 |
| `charngram`, augment | 60.90 | 47.33 |
| `vectors` GloVe-100, fallback | **66.35** | 50.88 |
| `vectors` GloVe-100, augment | 65.79 | 48.31 |
| `vectors` fastText on the corpus, fallback | — | 51.80 |
| `vectors` fastText on the corpus, augment | — | 45.97 |

On Reuters, pretrained vectors are worth **+5.1 points of unseen-label P@1**, and augment
mode also gives the best overall numbers (P@1 86.35 → 87.25, PSP@1 43.97 → 47.17). On npm
nothing helps. What separates them is whether the vector space covers the vocabulary: 97%
of Reuters label tokens have a GloVe vector against 77% for npm, whose JS jargon
(`webpack`, `eslint`, `serverless`) is absent from a 2014 Wikipedia corpus. The run prints
this coverage and warns below 90%.

Character n-grams help nowhere, and what they retrieve says why:

```
acq     -> cyacq(0.34), lt cyacq(0.27), provide cyacq(0.22), acquire(0.22)
bop     -> prop(0.11), crop(0.11), drop(0.11), stop(0.10)
housing -> housing(1.00), housing corp(0.70), housing starts(0.61), warehousing(0.60)
```

They capture *morphology*, which exact matching already covers, and are wrong on
*abbreviations*, which is where the gap is. Pretrained vectors fix some of that (`wpi` →
`cpi`, `wholesale`) but not all: `acq` and `bop` are rare enough that their GloVe vectors
are noise too.

Two implementation details mattered more than the choice of embedding: a name is only
embedded when *all* of its tokens are known (averaging over a partly missing name collapses
`middleware webpack` onto `middleware` and scores a spurious cosine of 1), and
`direct_fallback` decides whether fuzzy links only fill gaps or also sit alongside exact
matches. Augmenting is better when the vectors are good and worse when they are not.

## Label feature-bag expansion (off by default)

The direct map widens *retrieval*. This widens the label itself. Instead of

```
hiking  ->  {__label__7__hiking, 1_hiking}
```

each label also carries its nearest neighbours in the point vocabulary:

```
hiking  ->  {__label__7__hiking, 1_hiking, 1_trail, 1_outdoor}
```

The mechanism is different from fuzzy matching, and it is the reason this is worth doing:
`1_trail` is a label feature that *other, seen* labels already carry, so it already has a
**trained weight** in `W`. Adding it routes an unseen label through the learned channel
rather than only the untrained `knn` term.

```python
from zestxml import build_dataset
from zestxml.embed import glove_expander

build_dataset(..., label_expand=glove_expander("glove.6B.100d.txt", topk=2, min_sim=0.7))
```

`label_expand(names, name_tokens, vocab) -> {name: extra tokens}` is called once with the
whole label set and the point vocabulary — any source works, including an LLM. The default
`None` leaves the output byte-identical to a build without the hook, which a test enforces.

**It is worth a lot on one dataset and costs a lot on the other.** Unseen-label P@1,
`glove_expander` at k=2 / floor 0.7:

| | Reuters | npm |
| --- | --- | --- |
| control | 61.28 | **52.11** |
| expanded | **69.36** | 44.88 |
| all-label P@1 | 86.32 → 86.75 | 73.04 → 73.13 |
| seen P@1 | 95.08 → 95.08 | 74.97 → 75.09 |
| PSP@5 | 62.94 → **72.72** | 28.64 → 28.36 |

Both columns are end-to-end runs through `build_dataset` and `ZestXML`, control and variant
alike, at the reference configuration for each dataset.

The Reuters gain survived being attacked: it reproduces at seeds 0/1/2 with a delta of
+8.08 every time against a seed spread of 0.38, the two arms differ in `Yf.txt` and
`Y_Yf.txt` and nothing else (same unseen label set, same points, same shortlist size, same
epochs), and no test data enters the expansion. Two honest deductions: the k=2/0.7
operating point was picked on the test split over a 4-cell grid — all four cells beat
control on unseen P@1 (+4.3 / +8.1 / +4.3 / +4.5), so the direction is not a selection
artifact but the defensible expected gain is nearer **+5.7 to +7.6** than +8.08 — and the
seen-label cost is about **−0.1**, not the zero that seed 0 happens to show.

**What decides the sign is how widely an added feature is shared**, not vocabulary
coverage. `build_dataset` prints it:

```
expansion : 71 added features, 1.25 labels each on average (max 6); most shared: wheat x6, corn x6, ...
```

On Reuters the neighbours are domain nouns on ~1.25 labels each. On npm they are generic
English on 1.54 labels each with a tail to 19 — `1_example` lands on 19 labels,
`1_internet` on 13 — because GloVe's neighbourhood of a software keyword is prose, not
software. Those features cannot discriminate, and the unit-normalised `Y_Yf` row dilutes
the label's own token to pay for them. Note that coverage would have told you the opposite:
99% of npm's unseen label tokens are already in the point vocabulary, and it still loses.

Retrieval is not the explanation — npm's shortlist recall went slightly *up* (71.82% →
72.49%), so the loss is entirely in scoring. The obvious next knob is to weight expanded
features below the label's own tokens instead of at parity, or to expand only to tokens a
seen label already carries; neither is implemented.

`benchmarks/expansion_check.py` rebuilds a dataset through `build_dataset` both ways and
prints the two evaluations side by side, which is how the table above was produced.

### What did not work: a fused dense scoring term

The other obvious hybrid — adding `c * cos(enc(doc), enc(label))` as a third channel
beside the bilinear and `knn` terms, with `(a, b, c)` fit rather than guessed — was
implemented and measured, and **it loses on both datasets**. With the weights fit honestly
on held-out data: npm P@1 73.04 → 72.21 and unseen 52.11 → **5.27**; Reuters 86.35 → 84.17
and unseen 61.28 → 52.26. Even tuning the weights *on the test set* leaves npm below its
control (72.85 vs 73.04).

Two findings are worth keeping. The fitting is structurally broken for this problem: the
validation points are the classifier's own training points, so the search sees the bilinear
term as near-perfect (validation P@1 96.9 / 99.6), puts all the mass on it, and deletes the
`knn` term that is the entire zero-shot mechanism — hence the 5.27. Worse, a validation
shortlist built from training data contains **no unseen labels at all**, so no amount of
tuning on it can ever select for the thing being measured. Fixing this needs a validation
split held out *before* the classifier is trained, with unseen labels present in it.

And most of the one positive-looking row was not the dense term: deleting the dense channel
entirely and re-tuning only `score_alpha` on standardised channels recovers 8.11 of the
8.76 PSP@5 "gain". Per-point standardisation itself costs P@1 (86.35 → 84.47 at the same
0.9/0.1 mix), because the cross-candidate magnitude of the `knn` term is what carries
unseen labels. The residual signal is that `score_alpha=0.9` may be mistuned for PSP — a
property of the two existing channels, not of a new one.

## Validating on real data

The public GZXML datasets are large Google Drive downloads. Two smaller ones can be built
from scratch, both reproducing the generalized zero-shot setting by stripping a slice of
labels out of `trn_X_Y` while keeping them in `tst_X_Y` and `Y_Yf`.

```shell
pip install nltk scikit-learn
python -c "import nltk; nltk.download('reuters')"
python benchmarks/datasets/make_reuters.py GZXML-Datasets/GZ-Reuters-90
python benchmarks/datasets/make_npm.py GZXML-Datasets/GZ-NPM --cache npm_packages.jsonl
```

**GZ-Reuters-90** — 7769 train / 3019 test points, 90 labels (15 unseen), 36599 point
features; 16.1% of test positives belong to labels with no training example. Same
hyper-parameters, C++ against this implementation:

| | P@1 | P@3 | P@5 | nDCG@5 | PSP@1 | PSP@5 |
| --- | --- | --- | --- | --- | --- | --- |
| all labels, C++ | 85.56 | 33.71 | 21.25 | 87.24 | 41.17 | 56.82 |
| all labels, PyTorch | **86.42** | **34.23** | **21.68** | **88.68** | **43.76** | **62.15** |
| unseen only, C++ | 55.83 | 26.57 | 19.55 | 72.23 | 55.83 | 86.38 |
| unseen only, PyTorch | **59.96** | 26.57 | 19.51 | **73.72** | **59.96** | 86.21 |
| seen only, C++ | 95.08 | 36.61 | 22.64 | 96.83 | 86.57 | 93.68 |
| seen only, PyTorch | **95.19** | **36.70** | **22.75** | **97.15** | 86.12 | **93.80** |

Labels with no training example are retrieved at 59.96 P@1 purely from their text, which is
the behaviour the method exists for.

**GZ-NPM** — npm package descriptions tagged with their keywords, the same shape as the
paper's GZ-Amazon (tag an item from a long tail of textual tags). 25127/8376 points, 3223
labels of which 286 unseen, 4.9 tags per test point, 19.6% of test positives on unseen
labels.

| 3223 labels | P@1 | P@3 | P@5 | nDCG@5 | PSP@1 | PSP@5 |
| --- | --- | --- | --- | --- | --- | --- |
| all labels, C++ | **73.82** | **53.00** | **40.72** | **60.34** | **19.19** | 27.99 |
| all labels, PyTorch | 73.02 | 52.64 | 40.65 | 60.08 | 19.09 | **28.54** |
| unseen only, C++ | 50.69 | 27.47 | **17.94** | 54.87 | 50.69 | **56.17** |
| unseen only, PyTorch | **51.65** | **27.54** | 17.93 | **55.19** | **51.65** | 56.16 |

Here the two are a wash: the C++ is ~0.8 P@1 ahead on head labels, this one ~1 point ahead
on labels it has never seen. Both optimisers reach the same primal objective (462095
against 462297, 0.04% apart) but land at different points in it, the C++ with more
regularisation (reg 150695, data loss 311399) and this with a closer fit (reg 190906, data
loss 271391). Training for fewer epochs moves it less than a tenth of a point, so it is
where the optimum sits, not how long it is trained. Lower `bilinear_classifier_cost` for
the reference's balance.

The C++ was faster on CPU — 4.9s against 20s on Reuters, 23s against 52s on npm, 4 threads.
The sparse joins are written for GPU throughput and carry per-kernel overhead a CPU run
does not amortise. Use `device="cuda"` on a GPU box.

## Benchmarks against other methods

`benchmarks/RESULTS.md` has the full sweep — SPLADE-style learned sparse retrieval, OVA
linear, kNN, tf-idf centroid, BM25, dense probes, low-rank factorisation, label-graph
propagation, and Renee. The short version:

* **Per-label classifiers score exactly zero on unseen labels.** OVA, kNN and centroid all
  land on the evaluator's tie-break floor, because they have no mechanism to score a label
  with no training positive. This is the gap ZestXML exists to close, and it is total.
* **BM25 with zero training recovers 91% of ZestXML's unseen accuracy** — a reminder of how
  much of the zero-shot signal is plain lexical overlap.
* **Candidate-generation recall is the ceiling** (71.8% on npm) and resisted every variant.
* Renee, end-to-end on a GPU in 39 minutes, finished behind ZestXML's ~1 CPU minute on
  every metric at this data scale.

`benchmarks/colab/` has A100 notebooks; `benchmarks/colab_benchmark.sh` runs the same thing
from a shell.

## Tests

```shell
pytest tests -v
```

`tests/test_units.py` checks every sparse primitive against a dense reference, the dataset
conventions against a rebuilt directory, and the public API end to end. The C++ parity
suite that validated the original port has been removed along with the C++; its findings
are recorded above.

## Public datasets

Used in the paper, all downloadable [here](https://drive.google.com/file/d/1Cyi40UP9b527DiPrfJuvqmwm8OUUii0o/view?usp=sharing):
**GZ-EURLex-4.3K** (EU law page tagging), **GZ-Amazon-1M** (item to item recommendation),
**GZ-Wikipedia-1M** (Wikipedia page tagging).

<table>
  <tr> <td><b>Dataset</b></td> <td colspan="2" align="center"><b>Num Points</b></td> <td colspan="2" align="center"><b>Num Labels</b></td> <td colspan="2" align="center"><b>Num Features</b></td> </tr>
  <tr> <td></td> <td><b>Train</b></td> <td><b>Test</b></td> <td><b>Seen</b></td> <td><b>Unseen</b></td> <td><b>Point</b></td> <td><b>Label</b></td> </tr>
  <tr> <td><b>GZ-Eurlex-4.3K</b></td> <td>45,000</td> <td>6,000</td> <td>4,108</td> <td>163</td> <td>100,000</td> <td>24,316</td> </tr>
  <tr> <td><b>GZ-Amazon-1M</b></td> <td>914,179</td> <td>1,465,767</td> <td>476,381</td> <td>483,725</td> <td>1,000,000</td> <td>1,476,381</td> </tr>
  <tr> <td><b>GZ-Wikipedia-1M</b></td> <td>2,271,533</td> <td>2,705,425</td> <td>495,107</td> <td>776,612</td> <td>1,000,000</td> <td>1,438,196</td> </tr>
</table>

## Data format

All sparse matrices are stored in a plain-text row-major format:

* First line: `num_row num_column`.
* `num_row` lines follow, each a sparse row vector of space-separated `<index>:<value>`
  entries, 0-indexed. `[0, 0, 0.5, 0.4, 0, 0.2]` becomes `2:0.5 3:0.4 5:0.2`.

The files:

* **`Xf.txt`** — every feature of the document tf-idf representation, one per line. In the
  paper's datasets these are stemmed unigrams and bigrams; any feature set works.
* **`Yf.txt`** — the same for labels. Beyond unigrams and bigrams there is one feature
  unique to each label (`__label__<i>__<label-i-text>`), which gives the model label-specific
  parameters and helps on many-shot labels. `__parent__` features are specific to
  GZ-EURLex-4.3K, whose raw labels carry parent-concept information; ignore them elsewhere.
* **`(trn/tst/val)_X_Xf.txt`** — *documents × document-features* tf-idf.
* **`Y_Yf.txt`** — *labels × label-features*.
* **`trn_Y_Yf.txt`** — `Y_Yf` restricted to seen labels.
* **`(trn/tst/val)_X_Y.txt`** — *documents × labels* relevance.

## Cite

```bib
@InProceedings{Gupta21,
  author    = "Gupta, N. and Bohra, S. and Prabhu, Y. and Purohit, S. and Varma, M.",
  title     = "Generalized Zero-Shot Extreme Multi-label Learning",
  booktitle = "Proceedings of the ACM SIGKDD Conference on Knowledge Discovery and Data Mining",
  month     = "August",
  year      = "2021"
}
```
