"""Command line wrapper around :func:`zestxml.eval.report`.

    python tools/eval_xc.py Results/<dataset>/score_mat.bin GZXML-Datasets/<dataset>
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zestxml.eval import report  # noqa: E402

if __name__ == "__main__":
    print(sys.argv[1])
    report(sys.argv[1], sys.argv[2])
