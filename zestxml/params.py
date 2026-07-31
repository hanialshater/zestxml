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
    # ---- shortlist -----------------------------------------------------------
    "shortyK": "100",
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
    "lr": "0.05",  # Adam step size for the bilinear classifier
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
