"""The three-line API.

    from zestxml import build_dataset, ZestXML

    build_dataset("GZXML-Datasets/MyData", trn_texts, trn_labels, tst_texts, tst_labels)
    model = ZestXML("GZXML-Datasets/MyData", "Results/MyData").fit()
    model.predict()
    model.evaluate()

:class:`ZestXML` is a thin object over :mod:`zestxml.pipeline`: it fills in the eight file
paths from a dataset directory, keeps the model directory next to the results, and takes
hyper-parameters as keyword arguments instead of ``-name value`` strings. Every keyword is
one of :data:`zestxml.params.DEFAULTS`; an unknown one is an error rather than a silently
ignored typo. ``run_torch.py`` remains the command line front end and takes the same names.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

from .csr import CSR
from .eval import report
from .io import read_bin_smat
from .params import DEFAULTS, Params
from .pipeline import run_predict, run_xhtp_approx, run_xhtp_fine_tune

# the dataset files a run reads, as {parameter: filename}
DATA_FILES = {
    "trn_X_Xf": "trn_X_Xf.txt",
    "tst_X_Xf": "tst_X_Xf.txt",
    "Y_Yf": "Y_Yf.txt",
    "trn_X_Y": "trn_X_Y.txt",
    "tst_X_Y": "tst_X_Y.txt",
    "Xf": "Xf.txt",
    "Yf": "Yf.txt",
}

# tst_X_Y is not among them: it is only used to report shortlist recall, and predicting on
# data with no ground truth is a legitimate thing to want.
REQUIRED = [name for name in DATA_FILES if name != "tst_X_Y"]


class ZestXML:
    """A configured run over one dataset directory."""

    def __init__(
        self,
        data_dir: str,
        res_dir: str = "Results/run",
        model_dir: Optional[str] = None,
        **options,
    ):
        unknown = sorted(set(options) - set(DEFAULTS))
        if unknown:
            raise TypeError(
                "unknown parameter(s) %s -- see zestxml.params.DEFAULTS" % ", ".join(unknown)
            )
        for name, fname in DATA_FILES.items():
            path = os.path.join(data_dir, fname)
            if name not in options and os.path.exists(path):
                options[name] = path
        # Fail here rather than deep in the pipeline: an absent file otherwise leaves the
        # parameter at its "-" default and surfaces much later as FileNotFoundError: '-'.
        missing = [DATA_FILES[n] for n in REQUIRED if n not in options]
        if missing:
            hint = ("the directory does not exist" if not os.path.isdir(data_dir)
                    else "present: " + (", ".join(sorted(os.listdir(data_dir))[:8]) or "nothing"))
            raise FileNotFoundError(
                "%s is not a dataset directory -- missing %s (%s). Build one with "
                "zestxml.build_dataset()." % (data_dir, ", ".join(missing), hint)
            )

        self.data_dir = data_dir
        self.res_dir = res_dir
        self.model_dir = model_dir or os.path.join(res_dir, "model")
        options.setdefault("res_dir", res_dir)
        options.setdefault("model_dir", self.model_dir)
        self.params = Params({**DEFAULTS, **{k: str(v) for k, v in options.items()}})

    def __repr__(self) -> str:
        changed = {k: v for k, v in self.params.values.items()
                   if k not in DATA_FILES and v != DEFAULTS.get(k)}
        return "ZestXML(%r, %s)" % (self.data_dir, ", ".join("%s=%s" % kv for kv in changed.items()))

    # ------------------------------------------------------------------ stages
    def fit(self) -> "ZestXML":
        """Mine the sparsity pattern of ``W``, then train the bilinear classifier."""
        os.makedirs(self.model_dir, exist_ok=True)
        self.params.dump(os.path.join(self.res_dir, "params.txt"))
        run_xhtp_approx(self.params)
        run_xhtp_fine_tune(self.params)
        return self

    def predict(self) -> CSR:
        """Score the test points and return the sparse score matrix."""
        run_predict(self.params)
        return self.scores

    @property
    def scores(self) -> CSR:
        return read_bin_smat(os.path.join(self.res_dir, "score_mat.bin"))

    def evaluate(self, verbose: bool = True) -> Dict[str, Dict[str, float]]:
        """P@k / nDCG@k / PSP@k over all labels, and over unseen and seen labels apart."""
        return report(os.path.join(self.res_dir, "score_mat.bin"), self.data_dir, verbose=verbose)

    def run(self, verbose: bool = True) -> Dict[str, Dict[str, float]]:
        """fit + predict + evaluate."""
        self.fit()
        self.predict()
        return self.evaluate(verbose=verbose)
