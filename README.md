<img src="Resources/KDD_Logo.jpg" height="100" align="right"/>

# Generalized Zero-Shot Extreme Multi-Label Learning
This is the official codebase for [KDD 2021](https://www.kdd.org/kdd2021/) paper [Generalized Zero-Shot Extreme Multi-Label Learning](http://manikvarma.org/pubs/gupta21.pdf)
> [Nilesh Gupta](https://nilesh2797.github.io/), [Sakina Bohra](https://www.linkedin.com/in/sakina-bohra-aa46b174/?originalSubdomain=in), [Yashoteja Prabhu](https://vervenumen.github.io/), Saurabh Purohit, [Manik Varma](http://manikvarma.org/)

## Overview
Extreme Multi-label Learning ([`XML`](http://manikvarma.org/downloads/XC/XMLRepository.html)) involves assigning the subset of most relevant labels to a data point from extremely large set of label choices. An unaddressed challenge in XML is that of predicting unseen labels with no training points. 

Generalized Zero-shot XML (`GZXML`) is a paradigm where the task is to **tag a data point with the most relevant labels from a large universe of both seen and unseen labels**.

## Running the Code
```shell
# Build
make

# Download GZ-Eurlex-4.3K dataset
mkdir GZXML-Datasets
cd GZXML-Datasets
pip install gdown
gdown "https://drive.google.com/uc?id=1j27bQZol6gOQ7AATawShcF4jXJr3Venb"
tar -xvzf GZ-Eurlex-4.3K.tar.gz
cd -

# Train and predict ZestXML on GZ-Eurlex-4.3K dataset
./run_eurlex.sh train
./run_eurlex.sh predict

# Install dependencies of metrics.py
pip install -r requirements.txt
# Install pyxclib for evaluation
git clone https://github.com/kunaldahiya/pyxclib.git
cd pyxclib
python3 setup.py install --user
cd -

# Prints evaluation metrics
python metrics.py GZ-Eurlex-4.3K
```
## PyTorch Implementation
`zestxml/` is a full PyTorch port of the C++ code in `Source/`. It runs the same three
stages, takes the same flags, and reads/writes the same file formats, so it is a drop-in
replacement for `./run` and its output still works with `metrics.py`.

```shell
pip install -r requirements-torch.txt

# same interface as ./run -- add -device cuda to use a GPU
python run_torch.py -trn_X_Xf ... -Y_Yf ... -type all

# same interface as run_eurlex.sh
./run_eurlex_torch.sh train
./run_eurlex_torch.sh predict
python metrics.py GZ-Eurlex-4.3K
```

### How it maps onto the C++
| stage | C++ | PyTorch |
| --- | --- | --- |
| `xhtp_approx` | `create_Xf_Yf*` in `Source/helper.cpp`, `prod_for_jaccard` in `Source/mat.h` | `zestxml/pattern.py` |
| `xhtp_fine_tune` | `get_approx_shortlist`, `learn_bilinear_classifier` in `Source/zestxml.cpp` | `zestxml/model.py` |
| `predict` | `predict` in `Source/zestxml.cpp` | `zestxml/pipeline.py` |
| sparse matrices | `SMat` (`Source/mat.h`) | `zestxml/csr.py` |

The model is unchanged: a bilinear form `score(x, y) = x^T W y + b` whose `W`
(*num_Xf x num_Yf*) is confined to a mined sparsity pattern, so the free parameters are
one vector over that pattern's non-zeros. Three differences are deliberate:

* **Training.** The C++ flattens every shortlisted pair into the `|pairs| x |w|` "linear
  form" matrix and runs liblinear's dual coordinate descent. Here the same primal
  objective, `0.5*||w||^2 + sum_i C_i * loss(y_i * w.phi_i)` with squared hinge
  (`-bilinear_classifier_kind 0`) or logistic (`1`) loss, is minimised with Adam. The
  linear form is never materialised: scores contract the three sparse operands directly
  through a sorted-key join, so peak memory depends on the batch rather than on the
  label or feature count. `-bilinear_classifier_maxitr` is the epoch count, `-batch_size`
  the points per gradient step, and `-lr` the step size (linearly decayed to 0 over
  training). Each epoch prints the objective, which is directly comparable to the
  reference: on the synthetic dataset the defaults reach 43270 against dual coordinate
  descent's 43303, with the same train accuracy to within half a point. Raise `-lr` or
  `-bilinear_classifier_maxitr` if the printed objective is still falling at the end.
* **Shortlisting** is exact. `get_approx_shortlist` walks an inverted index with a
  shrinking threshold to approximate the top-`shortyK` labels; the port computes
  `X_Xf @ sparsity_pattern @ Y_Yf^T` in chunks and takes an exact top-k, which can only
  improve shortlist recall.
* **Batch sizing** is by cost, not by row count: `-max_elems` caps the non-zeros expanded
  per batch and `-dense_elems` the size of any dense working block. Lower them if you run
  out of memory, raise them for throughput.

Extra flags: `-device auto|cpu|cuda[:n]`, `-lr`, `-batch_size`, `-seed`, `-max_elems`,
`-dense_elems`, `-float64`. `-num_thread 0` lets torch pick its own thread count.

One more consistency fix: with `-bilinear_normalize 1` the C++ divides the linear-form
features by their norm while training but not while predicting; this port applies the
same normalisation in both. Every shipped run script sets it to `0`, where the two agree.

Flags kept for compatibility but inert: `-F` (only used by the approximate shortlist);
`-propensity_A`, `-propensity_B` and `-binary_relevance`, which are inert in the C++ too
because `create_Xf_Yf_map` binarises `trn_X_Y` after `ips_weight` has written to it.
`-bilinear_add_bias 1` is rejected rather than silently ignored; no run script uses it.

### Fuzzy label-to-document matching (off by default)
The direct map is the only bridge an unseen label has to the document vocabulary, and in
the reference it is *string equality*: `acq` never reaches `acquisition`. `-direct_map`
replaces equality with a similarity search over feature names:

```shell
-direct_map charngram                      # tf-idf over character n-grams, no downloads
-direct_map vectors -direct_vectors w2v.txt  # cosine in a word2vec/GloVe/fastText space
-direct_topk 3 -direct_min_sim 0.5 -direct_fallback 1
```

Links are weighted by `bs_direct_wt * cosine`, so a loose match counts for less than an
exact one, and `-direct_fallback 1` (the default) only searches for label features that
have no exact match, leaving correct matches undiluted.

**Results are mixed, and the pattern is instructive.** Unseen-label P@1:

| direct map | Reuters (97% vector coverage) | npm (77% coverage) |
| --- | --- | --- |
| `exact` (reference) | 61.28 | **52.11** |
| `charngram`, fallback | 61.28 | 51.76 |
| `charngram`, augment | 60.90 | 47.33 |
| `vectors` GloVe-100, fallback | **66.35** | 50.88 |
| `vectors` GloVe-100, augment | 65.79 | 48.31 |
| `vectors` fastText on the corpus, fallback | -- | 51.80 |
| `vectors` fastText on the corpus, augment | -- | 45.97 |

On Reuters, pretrained vectors are worth **+5.1 points of unseen-label P@1**, and the
augment mode also gives the best overall numbers (P@1 86.35 -> 87.25, PSP@1 43.97 ->
47.17). On npm nothing helps. What separates them is whether the vector space covers the
vocabulary: 97% of Reuters label tokens have a GloVe vector against 77% for npm, whose
JS jargon (`webpack`, `eslint`, `serverless`) is absent from a 2014 Wikipedia corpus.
The run prints this coverage and warns below 90%.

Character n-grams help nowhere, and looking at what they retrieve says why:

```
acq     -> cyacq(0.34), lt cyacq(0.27), provide cyacq(0.22), acquire(0.22)
bop     -> prop(0.11), crop(0.11), drop(0.11), stop(0.10)
housing -> housing(1.00), housing corp(0.70), housing starts(0.61), warehousing(0.60)
```

They capture *morphology*, which exact matching already covers, and are wrong on
*abbreviations*, which is where the gap is. Pretrained vectors fix some of that
(`wpi` -> `cpi`, `wholesale`) but not all: `acq` and `bop` are rare enough that their
GloVe vectors are noise too.

Two implementation details that turned out to matter more than the choice of embedding:
a name is only embedded when *all* of its tokens are known (averaging over a partly
missing name collapses `middleware webpack` onto `middleware` and scores a spurious
cosine of 1), and `-direct_fallback` decides whether fuzzy links only fill gaps or also
sit alongside exact matches. Augmenting is the better setting when the vectors are good
and the worse one when they are not.

### Benchmarking on a GPU
`colab/ZestXML_A100_benchmark.ipynb` runs both implementations on **GZ-Eurlex-4.3K** with
identical hyper-parameters and reports P@k / nDCG@k / PSP@k, wall time and peak GPU
memory. The same thing from a shell:

```shell
bash tools/colab_benchmark.sh GZ-Eurlex-4.3K          # both implementations
bash tools/colab_benchmark.sh GZ-Eurlex-4.3K torch    # skip the C++ baseline
```

It downloads the dataset, builds `./run`, trains and predicts with both, and evaluates
them. Every stage is chunked, so an out-of-memory failure is a knob rather than a wall:
lower `-max_elems` (non-zeros expanded per batch), `-dense_elems` (entries in a dense
working block) or `-batch_size`.

### Validating on a real dataset
The public GZXML datasets are large Google Drive downloads. For a quick end-to-end check
on real text, `tools/make_reuters.py` builds one from Reuters-21578, whose topics are
named in plain words ("grain", "money-supply") and therefore have genuine label text.
It reproduces the generalized zero-shot setting by stripping a slice of the topics out of
`trn_X_Y` while keeping them in `tst_X_Y` and `Y_Yf`, so those labels have no training
point and are reachable only through their own tokens.

```shell
pip install nltk scikit-learn
python -c "import nltk; nltk.download('reuters')"
python tools/make_reuters.py GZXML-Datasets/GZ-Reuters-90
# 7769 train / 3019 test points, 90 labels (15 unseen), 36599 point features
# 16.1% of test positives belong to labels with no training example

ARGS="-bs_count 20 -bs_alpha 0.02 -bs_direct_wt 0.8 -shortyK 50 -bilinear_classifier_cost 5 -bilinear_normalize 0"
python run_torch.py -trn_X_Xf GZXML-Datasets/GZ-Reuters-90/trn_X_Xf.txt ... -type all $ARGS
python tools/eval_xc.py Results/GZ-Reuters-90/score_mat.bin GZXML-Datasets/GZ-Reuters-90
```

`tools/eval_xc.py` computes P@k, nDCG@k and propensity scored PSP@k without needing
pyxclib, and breaks them down by whether a label was seen during training. Same
hyper-parameters, same data, C++ against this port:

| | P@1 | P@3 | P@5 | nDCG@5 | PSP@1 | PSP@5 |
| --- | --- | --- | --- | --- | --- | --- |
| all labels, C++ | 85.56 | 33.71 | 21.25 | 87.24 | 41.17 | 56.82 |
| all labels, PyTorch | **86.42** | **34.23** | **21.68** | **88.68** | **43.76** | **62.15** |
| unseen only, C++ | 55.83 | 26.57 | 19.55 | 72.23 | 55.83 | 86.38 |
| unseen only, PyTorch | **59.96** | 26.57 | 19.51 | **73.72** | **59.96** | 86.21 |
| seen only, C++ | 95.08 | 36.61 | 22.64 | 96.83 | 86.57 | 93.68 |
| seen only, PyTorch | **95.19** | **36.70** | **22.75** | **97.15** | 86.12 | **93.80** |

Labels with no training example are retrieved at 59.96 P@1 purely from their text, which
is the behaviour the method exists for.

Reuters only has 90 labels, so `tools/make_npm.py` builds a second, larger dataset with
the same structure: npm package descriptions tagged with their keywords, which is the
same shape of problem as the paper's GZ-Amazon (tag an item from a long tail of textual
tags). 25127/8376 points, **3223 labels** of which 286 unseen, 4.9 tags per test point,
19.6% of test positives on unseen labels.

```shell
python tools/make_npm.py GZXML-Datasets/GZ-NPM --cache npm_packages.jsonl
```

| 3223 labels | P@1 | P@3 | P@5 | nDCG@5 | PSP@1 | PSP@5 |
| --- | --- | --- | --- | --- | --- | --- |
| all labels, C++ | **73.82** | **53.00** | **40.72** | **60.34** | **19.19** | 27.99 |
| all labels, PyTorch | 73.02 | 52.64 | 40.65 | 60.08 | 19.09 | **28.54** |
| unseen only, C++ | 50.69 | 27.47 | **17.94** | 54.87 | 50.69 | **56.17** |
| unseen only, PyTorch | **51.65** | **27.54** | 17.93 | **55.19** | **51.65** | 56.16 |

Here the two are a wash: the C++ is ~0.8 P@1 ahead on head labels, this port ~1 point
ahead on labels it has never seen. That is not a defect on either side -- both optimisers
reach the same primal objective (462095 against 462297, 0.04% apart) but land at
different points in it, the C++ with more regularisation (reg 150695, data loss 311399)
and this port with a closer fit (reg 190906, data loss 271391). Training for fewer epochs
moves it less than a tenth of a point, so it is where the optimum sits, not how long it
is trained. Lower `-bilinear_classifier_cost` if you want the reference's balance.

The C++ is still faster on CPU: 4.9s against 20s on Reuters, 23s against 52s on the npm
dataset, on 4 threads. The sparse joins are written for GPU throughput and carry
per-kernel overhead a CPU run does not amortise. Add `-device cuda` on a GPU box.

### Tests
```shell
pytest tests -v
```
`tests/test_units.py` checks every sparse primitive against a dense reference.
`tests/test_cpp_parity.py` builds `./run`, runs it on a synthetic dataset, and asserts
that the mined pattern matches entry for entry, that the C++ model scored by this
implementation reproduces the C++ `bilinear_score_mat` / `knn_score_mat` / `score_mat`,
that the exact shortlist recalls at least as much as the approximate one, and that a
pattern mined here loads back into the C++ binary. It is skipped when no compiler is
available.

The remaining gap is a benchmark run on the real datasets: the numbers above come from
synthetic data, which has no signal to learn, so it validates agreement rather than
accuracy.

### Two reference behaviours this port does not reproduce
Both were found while diffing against `./run` on synthetic data, and both are bugs rather
than modelling choices, so the PyTorch code computes the intended quantity instead.

* **Dropped score contributions.** `prod_helper` (`Source/mat.h`) marks an accumulator
  slot as "seen" with `if (sum[id] == 0) indices.push_back(id)`. A label feature whose
  running sum is still exactly zero when a second point feature touches it is therefore
  pushed twice; `prod` then writes the real value for the first copy and sets
  `sum[id] = 0`, leaving the duplicate holding zero, and `sparse_prod` assigns
  `mask[yf]` entry by entry so the trailing zero wins and that label feature contributes
  nothing. Dual coordinate descent leaves many weights at exactly `0`, which is what
  makes a zero prefix reachable. On the synthetic run it silently changes 2 of 24000
  test scores. `tests/test_cpp_parity.py::cpp_dropped_pairs` identifies exactly which
  pairs are affected, and every other pair matches the C++ to 1e-5.
* **Ties at the top-k cut.** `prod_for_jaccard` truncates each row with `std::sort`,
  which is unstable, so when several feature pairs share the k-th best score the pattern
  it keeps is arbitrary. The parity test allows the two implementations to disagree on
  tied entries only.

Also note that the C++ binary divides by `dim/1000` when sizing a progress bar, so it
crashes with SIGFPE on matrices with fewer than 1000 rows; the synthetic data in
`tests/make_synthetic.py` stays above that.

## Public Datasets
Following Datasets were used in the paper for benchmarking `GZXML` algorithms (all datasets can be downloaded from [here](https://drive.google.com/file/d/1Cyi40UP9b527DiPrfJuvqmwm8OUUii0o/view?usp=sharing))
* **GZ-EURLex-4.3K**, Document Tagging of EU law pages
* **GZ-Amazon-1M**, Item to Item Recommendation of Amazon products
* **GZ-Wikipedia-1M**, Document Tagging of Wikipedia pages

Following are some statistics of these datasets:
<table>
  <tr> <td><b>Dataset</b></td> <td colspan="2" align="center"><b>Num Points</b></td> <td colspan="2" align="center"><b>Num Labels</b></td> <td colspan="2" align="center"><b>Num Features</b></td> </tr>
  <tr> <td></td> <td><b>Train</b></td> <td><b>Test</b></td> <td><b>Seen</b></td> <td><b>Unseen</b></td> <td><b>Point</b></td> <td><b>Label</b></td> </tr>
  <tr> <td><b>GZ-Eurlex-4.3K</b></td> <td>45,000</td> <td>6,000</td> <td>4,108</td> <td>163</td> <td>100,000</td> <td>24,316</td> </tr>
  <tr> <td><b>GZ-Amazon-1M</b></td> <td>914,179</td> <td>1,465,767</td> <td>476,381</td> <td>483,725</td> <td>1,000,000</td> <td>1,476,381</td> </tr>
  <tr> <td><b>GZ-Wikipedia-1M</b></td> <td>2,271,533</td> <td>2,705,425</td> <td>495,107</td> <td>776,612</td> <td>1,000,000</td> <td>1,438,196</td> </tr>
</table>

## Data Format
All sparse matrices are stored in text sparse matrix format, please refer to the text sparse matrix format subsection for more details. Following are the details of required files:
* **`Xf.txt`**: all features used in `tf-idf` representation of documents (`(trn/tst/val)_X_Xf`), `ith` line denotes `ith` feature in the tf-idf representation. In particular, for datasets used in the paper, it's the stemmed bigram and unigram features of documents but you can choose to have any set of features depending on your application.
* **`Yf.txt`**: similar to `Xf.txt` it represents features of all labels. In addition to unigrams and bigrams, we also add a unique feature specific to each label (represented by `__label__<i>__<label-i-text>`, this feature will only be present in `ith` label's features), this allows the model to have label specific parameters and helps it to do well on many-shot labels. Features with `__parent__` in them are only specific to the `GZ-EURLex-4.3K` dataset because raw labels in this dataset have some additional information about parent concepts of each label, you can safely choose to ignore these features for any other/new dataset.
* **`(trn/tst/val)_X_Xf.txt`**: sparse matrix (*documents x document-features*) representing `tf-idf` feature matrix of *(trn/tst/val)* input documents.
* **`Y_Yf.txt`**: similar to `(trn/tst/val)_X_Xf.txt` but for labels, this is the sparse matrix (*labels x label-features*) representing `tf-idf` feature matrix of labels.
* **`trn_Y_Yf.txt`**: similar to `Y_Yf.txt` but contains features for only the seen labels (can be interpreted as `Y_Yf[seen-labels]`)
* **`(trn/tst/val)_X_Y.txt`**: sparse matrix (*documents x labels*) representing *(trn/tst/val)* document-label relevance matrix.

### Text sparse matrix format
This is a plain-text row-major representation of a sparse matrix. Following are the details of the format :
- The first line in this format is two space separated integers denoting the dimensions of the matrix (i.e. `num_row` `num_column`)
- `num_row` lines follow the first line and each line represents a sparse row vector
- a sparse row vector is represented as space separated non zero entries of the vector, an entry in the vector is represented as `<index>:<value>`. For example if the vector is `[0, 0, 0.5, 0.4, 0, 0.2]` then its sparse vector text representation is `2:0.5 3:0.4 5:0.2` (NOTE : the indexing starts from 0)
- You can check `GZ-Eurlex-4.3K/trn_X_Xf.txt` for sample example of a sparse matrix format

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
