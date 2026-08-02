"""Structured hard pursuit over semantic-ID prefix blocks: the ablation.

    python benchmarks/pursuit.py GZXML-Datasets/GZ-NPM glove.6B.100d.txt
    python benchmarks/pursuit.py GZXML-Datasets/GZ-Reuters-90 glove.6B.100d.txt --budgets 0.9,0.7,0.5

Six arms, all sharing one mined pattern, one shortlist and one training budget -- only the
mask over the weight vector changes:

* **control**              no pruning at all
* **cosine prune**         the shipped one-shot pre-training prune (``-prune_min_sim``)
* **pursuit, mild**        block pruning to a high budget
* **pursuit, aggressive**  block pruning to a low budget
* **unstructured**         same budget and schedule, one block per slot -- so it ranks
                           *entries*, which is plain magnitude pruning
* **label-side only**      blocks keyed by the label prefix alone, ignoring which document
                           word the link came from
* **xf x prefix**          blocks keyed by the raw ``xf`` and the label prefix, which on
                           Reuters gives 257634 blocks over 284540 slots

The three controls are what make this a test rather than a demo. If unstructured matches
pursuit, the semantic IDs are decoration. ``xf x prefix`` is expected to match unstructured
almost exactly, because at 1.1 slots per block the two are the same operation -- it is in
here to show that, not to win. All are reported whether or not they are flattering.

Quality is plotted against *active* parameters, which is the axis the whole idea is about.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zestxml import ZestXML  # noqa: E402

CONFIG = dict(bs_alpha=0.02, bs_direct_wt=0.8, bilinear_classifier_cost=5,
              bilinear_normalize=0, bilinear_classifier_maxitr=20, num_thread=0)


def arms(vectors, budgets, min_sim):
    yield "control", {}
    if min_sim > 0:
        yield f"cosine prune {min_sim}", dict(prune_vectors=vectors, prune_min_sim=min_sim)
    for b in budgets:
        yield f"pursuit {b:g}", dict(pursuit_vectors=vectors, pursuit_budget=b)
    lo = min(budgets) if budgets else 0.5
    # three controls, all at the most aggressive budget, where a difference should show
    yield f"unstructured {lo:g}", dict(pursuit_vectors=vectors, pursuit_budget=lo,
                                       pursuit_levels=1, pursuit_codebook=1)
    yield f"label-side only {lo:g}", dict(pursuit_vectors=vectors, pursuit_budget=lo,
                                          pursuit_side="y")
    yield f"xf x prefix {lo:g}", dict(pursuit_vectors=vectors, pursuit_budget=lo,
                                      pursuit_side="xf_y")
    # Sparsity driven by the block budget alone, with the slot cap left non-binding. The
    # arms above all hit their slot cap exactly, which means the |w| ranking decided the
    # mask and the tree only trimmed the edges -- these are the ones where the structure
    # has to do the work on its own.
    for b in budgets:
        yield f"blocks only {b:g}", dict(pursuit_vectors=vectors, pursuit_budget=1.0,
                                         pursuit_level_budgets=f"{b},{b},{b}")


def main(data_dir, vectors, budgets, min_sim, out="Results", **overrides):
    config = {**CONFIG, **overrides}
    tag = os.path.basename(data_dir.rstrip("/"))
    rows = []
    for name, extra in arms(vectors, budgets, min_sim):
        slug = name.replace(" ", "-").replace(".", "")
        print("\n" + "=" * 78 + f"\n=== {name}\n" + "=" * 78)
        # a fresh model dir per arm: the pattern, the seen-labels cache and the classifier
        # all live there, and sharing one across arms is how a whole npm run came back wrong
        res = f"{out}/Pursuit-{tag}/{slug}"
        model = ZestXML(data_dir, res, **config, **extra)
        metrics = model.run(verbose=False)
        rows.append((name, metrics, active_params(res)))

    print("\n" + "=" * 96)
    print("%-22s %7s %7s %7s %9s %9s %12s" % (
        "arm", "P@1", "P@5", "PSP@5", "unseen", "seen", "active params"))
    for name, m, n in rows:
        unseen = m.get("unseen only", {}).get("P@1", float("nan"))
        seen = m.get("seen only", {}).get("P@1", float("nan"))
        print("%-22s %7.2f %7.2f %7.2f %9.2f %9.2f %12s" % (
            name, m["all labels"]["P@1"], m["all labels"].get("P@5", float("nan")),
            m["all labels"]["PSP@5"], unseen, seen,
            f"{n:,}" if n else "-"))
    plot(rows, f"{out}/Pursuit-{tag}/quality_vs_params.png")
    return rows


def active_params(res_dir):
    """Non-zero weights in the trained classifier -- the number the sparsity claim is about."""
    import numpy as np

    path = f"{res_dir}/model/bilinear_clf.bin"
    if not os.path.exists(path):
        return 0
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from zestxml.io import read_bin_vec
    return int((np.asarray(read_bin_vec(path)) != 0).sum())


def plot(rows, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed, skipping the plot)")
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, m, n in rows:
        if not n:
            continue
        ax.scatter(n, m["all labels"]["P@1"], s=60)
        ax.annotate(name, (n, m["all labels"]["P@1"]), fontsize=8,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("active parameters")
    ax.set_ylabel("P@1")
    ax.set_title("quality vs sparsity")
    ax.grid(alpha=0.3)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"\nplot written to {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("vectors", help="GloVe/fastText text file, or a sentence-transformers name")
    ap.add_argument("--budgets", default="0.75,0.5",
                    help="comma-separated fractions of prunable slots to keep")
    ap.add_argument("--min_sim", type=float, default=0.30,
                    help="cosine floor for the shipped one-shot prune arm; 0 skips it")
    ap.add_argument("--out", default="Results")
    ap.add_argument("overrides", nargs="*", help="key=value run parameters")
    a = ap.parse_args()
    main(a.data_dir, a.vectors, [float(b) for b in a.budgets.split(",")], a.min_sim,
         a.out, **dict(o.split("=", 1) for o in a.overrides))
