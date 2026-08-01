"""End to end: build a dataset from raw text, train, evaluate.

    python examples/quickstart.py

Everything below is real, runnable code. Replace :func:`load_data` with your own and the
rest works unchanged -- the shape it returns (a list of strings and a list of label-name
lists, per split) is the whole interface.

The toy data here is shaped like an interest-mining problem: each "customer" is the
concatenated titles of items they interacted with, and the labels are interest names.
"""

import os
import random
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from zestxml.dataset import build_dataset, select_unseen_labels  # noqa: E402
from zestxml.params import Params  # noqa: E402
from zestxml.pipeline import run_predict, run_xhtp_approx, run_xhtp_fine_tune  # noqa: E402

DATA_DIR = "GZXML-Datasets/Quickstart"
RES_DIR = "Results/quickstart"


# --------------------------------------------------------------------------- #
# 1. your data
# --------------------------------------------------------------------------- #
def load_data(seed=0):
    """Return (texts, labels) for train and test.

    texts  : one string per point   -- e.g. a customer's item titles, concatenated
    labels : list of label NAMES per point -- e.g. the interests an LLM emitted

    Label names matter: their tokens are what lets a label with no training example be
    predicted at all, so keep them descriptive ("trail running" beats "cluster_47").
    """
    rng = random.Random(seed)
    interests = {
        "hiking": ["hiking boots", "trail backpack", "trekking poles", "waterproof jacket"],
        "photography": ["camera lens", "tripod stand", "camera bag", "memory card"],
        "coffee": ["coffee grinder", "espresso machine", "pour over kettle", "coffee beans"],
        "cycling": ["bike helmet", "cycling jersey", "bike lights", "chain lubricant"],
        "cooking": ["chef knife", "cast iron skillet", "cutting board", "mixing bowl"],
        "gaming": ["mechanical keyboard", "gaming mouse", "headset stand", "monitor arm"],
        "gardening": ["pruning shears", "garden gloves", "watering can", "seed trays"],
        "running": ["running shoes", "sports watch", "hydration belt", "compression socks"],
    }
    names = sorted(interests)

    texts, labels = [], []
    for _ in range(3000):
        picked = rng.sample(names, rng.randint(1, 3))
        items = [rng.choice(interests[p]) for p in picked for _ in range(rng.randint(2, 4))]
        rng.shuffle(items)
        texts.append(" ".join(items))
        labels.append(picked)

    split = int(0.8 * len(texts))
    return texts[:split], labels[:split], texts[split:], labels[split:]


# --------------------------------------------------------------------------- #
# 2. build the dataset directory
# --------------------------------------------------------------------------- #
def build(trn_texts, trn_labels, tst_texts, tst_labels, zero_shot=True):
    label_names = sorted({l for row in trn_labels + tst_labels for l in row})
    index = {n: i for i, n in enumerate(label_names)}

    unseen = None
    if zero_shot:
        # hold some labels out of training entirely: they keep their features and their
        # test positives, so they are zero-shot rather than simply absent
        unseen = select_unseen_labels(
            [[index[l] for l in row] for row in trn_labels],
            [[index[l] for l in row] for row in tst_labels],
            len(label_names), stride=4, min_test=5,
        )
        print("held out of training:", sorted(label_names[i] for i in unseen))

    return build_dataset(
        DATA_DIR,
        trn_texts=trn_texts, trn_labels=trn_labels,
        tst_texts=tst_texts, tst_labels=tst_labels,
        label_names=label_names, unseen=unseen,
    )


# --------------------------------------------------------------------------- #
# 3. train and predict
# --------------------------------------------------------------------------- #
def train_and_predict():
    """The same three stages `run_torch.py -type all` runs, called directly."""
    argv = []
    for key, value in {
        "trn_X_Xf": f"{DATA_DIR}/trn_X_Xf.txt", "tst_X_Xf": f"{DATA_DIR}/tst_X_Xf.txt",
        "Y_Yf": f"{DATA_DIR}/Y_Yf.txt", "trn_X_Y": f"{DATA_DIR}/trn_X_Y.txt",
        "tst_X_Y": f"{DATA_DIR}/tst_X_Y.txt", "Xf": f"{DATA_DIR}/Xf.txt",
        "Yf": f"{DATA_DIR}/Yf.txt",
        "res_dir": RES_DIR, "model_dir": f"{RES_DIR}/model",
        "device": "auto",            # cuda when available
        "shortyK": "8",              # candidates per point; with few labels, score them all
        "bs_count": "20",            # label features kept per point feature when mining W
        "bs_direct_wt": "0.8",       # weight of an exact label-token to point-token match
        "bilinear_classifier_cost": "5",
        "bilinear_classifier_maxitr": "20",
        "bilinear_normalize": "0",
        "num_thread": "0",
    }.items():
        argv += ["-" + key, value]

    params = Params.parse(argv)
    run_xhtp_approx(params)       # stage 1: mine the sparsity pattern of W
    run_xhtp_fine_tune(params)    # stage 2: shortlist + train the bilinear classifier
    run_predict(params)           # stage 3: score the test points


# --------------------------------------------------------------------------- #
# 4. evaluate
# --------------------------------------------------------------------------- #
def evaluate():
    subprocess.run(
        [sys.executable, "tools/eval_xc.py", f"{RES_DIR}/score_mat.bin", DATA_DIR],
        cwd=ROOT, check=True,
    )


if __name__ == "__main__":
    os.chdir(ROOT)
    trn_texts, trn_labels, tst_texts, tst_labels = load_data()
    print(f"loaded {len(trn_texts)} train / {len(tst_texts)} test points\n")

    build(trn_texts, trn_labels, tst_texts, tst_labels)
    print()
    train_and_predict()
    print()
    evaluate()
