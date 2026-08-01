"""ZestXML in PyTorch -- Generalized Zero-Shot Extreme Multi-label Learning.

    from zestxml import build_dataset, ZestXML

    build_dataset("GZXML-Datasets/MyData", trn_texts, trn_labels, tst_texts, tst_labels)
    ZestXML("GZXML-Datasets/MyData", "Results/MyData").run()

A label is a *bag of text features*, not an id, so the parameters are shared through the
label-feature vocabulary and a label with no training example is still scoreable. Every
stage is batched sparse tensor arithmetic, so the whole pipeline runs on a GPU and the
classifier is trained by gradient descent on the primal objective.
"""

from .api import ZestXML
from .csr import CSR
from .dataset import build_dataset, select_unseen_labels
from .eval import report
from .model import BilinearClassifier, BilinearPattern, get_shortlist, pair_scores, pair_targets
from .params import Params
from .pattern import build_sparsity_pattern, union_pattern
from .pipeline import main, run_predict, run_xhtp_approx, run_xhtp_fine_tune

__all__ = [
    "CSR",
    "BilinearClassifier",
    "BilinearPattern",
    "Params",
    "ZestXML",
    "build_dataset",
    "build_sparsity_pattern",
    "get_shortlist",
    "main",
    "pair_scores",
    "pair_targets",
    "report",
    "run_predict",
    "run_xhtp_approx",
    "run_xhtp_fine_tune",
    "select_unseen_labels",
    "union_pattern",
]
