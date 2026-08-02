"""Command line parameters, accepting the same ``-name value`` flags as ``run.cpp``."""

from __future__ import annotations

import os
import sys
from typing import Dict, List

DEFAULTS: Dict[str, str] = {
    # ---- data ----------------------------------------------------------------
    "trn_X_Xf": "-",
    "tst_X_Xf": "-",
    "trn_X_Y": "-",
    "tst_X_Y": "-",
    "Y_Yf": "-",
    "Xf": "-",
    "Yf": "-",
    "res_dir": "Results",
    "model_dir": "Results/model",
    "type": "-",
    # ---- sparsity pattern ----------------------------------------------------
    "bs_count": "10",
    "bs_threshold": "0",
    "bs_alpha": "0.2",
    "bs_direct_wt": "0.2",
    "sparsity_pattern_file": "-",
    "direct_map": "exact",  # exact | charngram | vectors
    "direct_fallback": "1",  # fuzzy-match only label features with no exact match
    "direct_topk": "3",  # fuzzy links kept per label feature
    "direct_min_sim": "0.5",  # cosine floor for a fuzzy link
    "direct_vectors": "-",  # word2vec/GloVe/fastText text file for -direct_map vectors
    # ---- semantic pruning of the mined pattern -------------------------------
    "prune_vectors": "-",  # vectors/model used to score how related a mined (xf, yf) pair is
    "prune_min_sim": "0.0",  # drop pattern entries below this cosine; 0 disables pruning
    # structured hard pursuit: prune whole semantic-ID prefix blocks of the pattern during
    # training, on learned weight magnitude, with ancestor closure and gradient revival
    "pursuit_vectors": "-",  # vectors/model used to give each label feature a code path
    "pursuit_budget": "0.0",  # fraction of prunable slots to keep; 0 disables pursuit
    "pursuit_levels": "3",
    "pursuit_codebook": "64",
    "pursuit_interval": "4",  # prune every this many epochs
    "pursuit_explore": "0.3",  # drop-and-grow rate, cosine-decayed to zero
    "pursuit_level_budgets": "-",  # optional per-level block caps, e.g. "32,512,4096"
    "pursuit_side": "xy",  # block key: "xy" both sides coded, "y" label side only,
                           # "xf_y" raw xf x label prefix (degenerate, kept as a control)
    "pursuit_keep_identity": "1",  # exempt __label__ features (the per-label residual)
    # ---- shortlist -----------------------------------------------------------
    "shortyK": "100",
    "shortlist_file": "-",  # test-time candidate set, instead of generating one
    "trn_shortlist_file": "-",  # training candidate set; the two have different shapes, so
    # training on semantically-retrieved candidates needs both (see benchmarks/rq_shortlist.py)
    "F": "10",
    # ---- classifier ----------------------------------------------------------
    "bilinear_classifier_kind": "0",
    "bilinear_classifier_maxitr": "20",
    "bilinear_classifier_cost": "1.0",
    "bilinear_classifier_pos_wt": "1.0",
    "bilinear_normalize": "1",
    "bilinear_add_bias": "0",
    "binary_relevance": "1",
    "propensity_A": "0.55",
    "propensity_B": "1.5",
    "score_alpha": "0.9",
    # ---- torch specific ------------------------------------------------------
    "device": "auto",  # auto | cpu | cuda | cuda:0 | ...
    "num_thread": "1",  # 0 -> let torch decide
    "lr": "0.2",  # Adam step size for the bilinear classifier
    "batch_size": "256",  # points per gradient step
    "seed": "0",
    "max_elems": str(1 << 24),  # cap on non-zeros expanded per batch
    "dense_elems": str(1 << 24),  # cap on entries of a dense working block
    "float64": "0",
}

# accepted for command line compatibility with run.cpp but without effect here
INERT = {
    "F": "only used by the C++ approximate shortlist; this port shortlists exactly",
    "propensity_A": "propensity weights are inert in the reference too (see README)",
    "propensity_B": "propensity weights are inert in the reference too (see README)",
    "binary_relevance": "propensity weights are inert in the reference too (see README)",
    "seen_labels": "seen labels are derived from trn_X_Y / model_dir/seen_labels.txt",
}


class Params:
    def __init__(self, values: Dict[str, str]):
        self.values = values

    # -------------------------------------------------------------- accessors
    def __contains__(self, name: str) -> bool:
        return name in self.values

    def str(self, name: str) -> str:
        return self.values[name]

    def int(self, name: str) -> int:
        return int(self.values[name])

    def float(self, name: str) -> float:
        return float(self.values[name])

    def bool(self, name: str) -> bool:
        v = self.values[name].strip().lower()
        return v not in ("0", "false", "no", "")

    def path(self, name: str) -> str:
        return self.values[name]

    def given(self, name: str) -> bool:
        return self.values.get(name, "-") != "-"

    # ------------------------------------------------------------------- setup
    @staticmethod
    def parse(argv: List[str]) -> "Params":
        values = dict(DEFAULTS)
        i = 0
        while i < len(argv):
            tok = argv[i]
            if tok.startswith("-") and not _is_number(tok):
                name = tok[1:]
                if i + 1 < len(argv) and not (argv[i + 1].startswith("-") and not _is_number(argv[i + 1])):
                    values[name] = argv[i + 1]
                    i += 2
                    continue
                sys.stderr.write("Invalid argument : no value provided for param %s\n" % name)
            i += 1
        return Params(values)

    def dump(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            for k in sorted(self.values):
                f.write("%s %s\n" % (k, self.values[k]))


def _is_number(tok: str) -> bool:
    try:
        float(tok)
        return True
    except ValueError:
        return False
