"""xDR -- expected decision regret."""

# Import order workaround, not a dependency of anything below: on this
# Windows/Python 3.12 environment, `lightgbm.Booster.predict()` on a *small*
# batch (a few dozen to a few thousand rows -- exactly the size of a single
# moment's counterfactual candidates) segfaults with
# "OSError: exception: access violation reading 0x0000000000000000" if
# `lightgbm` is imported before any of scikit-learn's compiled/OpenMP-linked
# extensions (`sklearn.neighbors` reproduces it reliably; likely a native
# OpenMP-runtime initialization-order conflict between the two libraries'
# bundled runtimes). Predicting on large batches (the ~16k-row test split in
# evaluation/report.py) was unaffected either way, which is why this was not
# caught until the counterfactual layer (M8) started calling `predict()` on
# per-moment candidate batches. Importing `sklearn.neighbors` here, before
# any submodule of this package gets a chance to `import lightgbm`, fixes it
# process-wide: `xdr/__init__.py` runs before any `xdr.*` submodule import,
# regardless of which one a script imports first. Verified: removing this
# line reproduces the crash; keeping it, the identical predict call succeeds
# every time. See docs/DECISIONS.md.
import sklearn.neighbors  # noqa: F401

__version__ = "0.1.0"
