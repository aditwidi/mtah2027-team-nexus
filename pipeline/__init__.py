"""The possession pipeline, grouped by role.

    data/    io.py, features.py                   provide the assets, build the per-frame matrix
    model/   emissions.py, decode.py, learn.py     evidence -> labels
    checks/  diag.py, verify.py, render.py         label-free, oracle and visual verification
    cli/     submit.py, ablate.py                  the two commands that write files

Dependencies point one way: cli -> checks/model -> data. Nothing here imports from a sibling
group in the other direction, so no cycles are possible.

Importing anything from this package puts the repository root on ``sys.path``, which is what makes
the top-level ``games.py`` of the upstream repository importable without an install step. Every
module reaches the repository root through ``io.ROOT``.
"""

import os
import sys

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PACKAGE_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
