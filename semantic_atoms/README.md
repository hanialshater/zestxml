# Sparse semantic atoms for ZestXML

Tests the claim that **a sparse model can be semantic**: ZestXML's machinery
(bilinear scoring, sparse shortlisting) is representation-agnostic — it needs a
basis that is high-dimensional, sparse, and *shared between the document and
label sides*, not necessarily a lexical one. Semantic-ID (RQ-VAE) codes are one
sparse semantic basis; a flat overcomplete k-sparse code is another. This
experiment compares them inside the unmodified ZestXML binary.

## Setup

Zero-shot dataset built from Reuters-21578 (via NLTK — chosen because it is
fetchable in a sandboxed environment; every script is dataset-agnostic and runs
unchanged on GZ-Eurlex-4.3K etc., see below): 7,105 train / 3,019 test docs,
90 labels of which **25 are held out as unseen** (zero training column —
ZestXML's `remove_test_labels` then excludes them from training automatically).

Three feature sets, identical hyperparameters and label text everywhere:

1. **lexical** — stemmed unigram+bigram tf-idf, per-label unique features
   (standard ZestXML recipe).
2. **+ atoms** — lexical ⊕ k-sparse semantic atoms: corpus tf-idf → LSA (256d)
   → tied-weight k-sparse autoencoder (512 atoms, k=16, unit-norm decoder
   columns). Atom activations become features `#atom<j>` on **both** the Xf and
   Yf side (no underscore in the token, so ZestXML's direct Xf↔Yf map —
   `create_Xf_Yf_map_direct`, which strips a Yf token through its first `_` —
   matches them exactly). Atom block l2-normalized per row, scaled 0.3 relative
   to the lexical block. Nothing pretrained/external: the basis is learned from
   the training corpus + label texts only.
3. **+ RQ codes** — same LSA embeddings, but quantized semantic-ID style:
   residual k-means, 4 levels × 32 centroids, 4 active code features per
   doc/label, same both-sides placement and the same 0.3 weight. The only
   difference vs (2) is the *shape* of the sparse basis: a tree path instead of
   a flat overcomplete code.

## Results (GZ-Reuters-90)

| unseen labels (374 docs)   | P@1   | P@3   | P@5   | R@5   | R@10  |
|----------------------------|-------|-------|-------|-------|-------|
| lexical baseline           | 74.87 | 28.79 | 17.27 | 78.25 | 78.25 |
| + RQ codes (semantic IDs)  | 76.20 | 31.11 | 18.88 | 84.43 | 84.43 |
| + sparse atoms             | **78.07** | **33.16** | **19.95** | **88.79** | **89.59** |

| seen / overall             | seen P@1 | overall P@1 |
|----------------------------|----------|-------------|
| lexical baseline           | 94.56    | 87.68       |
| + RQ codes                 | 94.20    | 87.21       |
| + sparse atoms             | 94.20    | 87.38       |

Takeaways:

- **Semantic sparse features help zero-shot, full stop.** Unseen recall@10
  jumps from 78.3 → 89.6 (+11.3) with atoms, at a ~0.3pt cost on seen P@1.
  The lexical model's unseen recall saturates (R@5 = R@10): what it can't
  lexically match it never surfaces at any depth. Atoms fix precisely that.
- **The basis shape matters.** RQ-style codes through the *identical* harness
  also help (their recall saturates at 84.4) but lose to flat overcomplete
  atoms on every unseen metric. Tree-path codes are too coarse per level and
  non-compositional; k-sparse atoms decompose an unseen label into many
  trained, shared components.
- **Both-sides placement is load-bearing.** The RQ variant here already fixes
  the usual semantic-ID mistake (codes only on the label side). Even in its
  strongest form, it trails the atoms.
- **Atom weight is a real knob**: at weight 1.0 atoms boost recall but hurt
  unseen P@1 (within-cluster ties — gold/silver/platinum share atoms — drown
  the lexical tie-breaker); 0.3 gets both. Sweep it on a new dataset.

Caveats: 90 labels is far from extreme scale; the unseen-P@1 deltas are within
~2σ for n=374 (the recall deltas are not); Reuters labels are lexically easy,
which biases *against* the semantic features — the gap should widen on
datasets with paraphrased/abstract label text.

## Running

```shell
make                                # build ZestXML
pip install numpy scipy scikit-learn nltk torch tqdm
python3 -c "import nltk; [nltk.download(p) for p in ['reuters','stopwords']]"
./semantic_atoms/run_experiment.sh
```

## Other datasets (e.g. GZ-Eurlex-4.3K)

`build_atoms.py`, `build_rq_codes.py` and `evaluate.py` are dataset-agnostic:
they need a GZXML-format dir plus `raw/{trn_X,tst_X,Y}.txt` (one text per line,
as shipped with the GZ datasets) and an optional `unseen_labels.txt` (one label
index per line; for GZ datasets, derive it as the columns of `trn_X_Y` with
zero support). For Eurlex-scale label sets, scale the dictionary up, e.g.
`--n-atoms 4096 --k 16`, and re-sweep `--atom-wt`.
