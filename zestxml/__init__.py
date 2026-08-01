"""ZestXML in PyTorch -- Generalized Zero-Shot Extreme Multi-label Learning.

A port of the C++ reference implementation in ``Source/`` that keeps the algorithm and
the on-disk formats but expresses every stage as batched sparse tensor operations, so the
whole pipeline runs on a GPU and the classifier is trained by gradient descent on the
primal objective instead of liblinear's dual coordinate descent.
"""

from .csr import CSR
from .dataset import build_dataset, select_unseen_labels
from .model import BilinearClassifier, BilinearPattern, get_shortlist, pair_scores, pair_targets
from .params import Params
from .pattern import build_sparsity_pattern, union_pattern
from .pipeline import main, run_predict, run_xhtp_approx, run_xhtp_fine_tune

__all__ = [
    "CSR",
    "BilinearClassifier",
    "BilinearPattern",
    "Params",
    "build_dataset",
    "build_sparsity_pattern",
    "select_unseen_labels",
    "get_shortlist",
    "main",
    "pair_scores",
    "pair_targets",
    "run_predict",
    "run_xhtp_approx",
    "run_xhtp_fine_tune",
    "union_pattern",
]
