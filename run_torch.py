#!/usr/bin/env python3
"""PyTorch entry point.

    python run_torch.py -trn_X_Xf ... -type all
"""

import sys

from zestxml.pipeline import main

if __name__ == "__main__":
    sys.exit(main())
