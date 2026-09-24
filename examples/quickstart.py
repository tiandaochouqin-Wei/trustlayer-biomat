"""Quickstart: wrap any scikit-learn regressor with the trust layer (synthetic data, ~5 s).

A property y depends on two composition variables. The model is trained and calibrated on a
"conventional" region (x1 in [0, 1]) and then asked about a "design" region (x1 in [1.5, 2.5])
it has never seen. Run from the repository root:  python examples/quickstart.py
"""
import os
import sys

import numpy as np
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
from trustlayer import TrustLayer  # noqa: E402

rng = np.random.default_rng(0)


def draw(n, design=False):
    X = rng.uniform(0, 1, size=(n, 2))
    if design:
        X[:, 1] = rng.uniform(1.5, 2.5, size=n)
    y = 10 * X[:, 0] + 3 * np.sin(2.5 * X[:, 1]) + rng.normal(0, 0.3, n)
    return X, y


X_tr, y_tr = draw(600)
X_cal, y_cal = draw(300)
X_src, y_src = draw(400)                      # held-out conventional candidates
X_des, y_des = draw(400, design=True)         # design-region candidates
X_rec, y_rec = draw(30, design=True)          # a budget of m = 30 target measurements

layer = TrustLayer(RandomForestRegressor(300, random_state=0), alpha=0.10)
layer.fit(X_tr, y_tr)
layer.calibrate(X_cal, y_cal)


def coverage(X, y, checked=True):
    """checked=False -> plain source-calibrated split conformal (all-False flags)."""
    flags = None if checked else np.zeros(len(X), bool)
    lo, hi = layer.interval(X, flags=flags)
    inf = float(np.mean(~np.isfinite(hi - lo)))
    return float(np.mean((y >= lo) & (y <= hi))), inf


def flagged(X):
    return float(np.mean(layer.shift_check(X)[1]))     # returns (p-values, flags)


print("nominal coverage                                 0.90")
print(f"conventional region, source-calibrated           {coverage(X_src, y_src, False)[0]:.2f}")
print(f"design region,       source-calibrated           {coverage(X_des, y_des, False)[0]:.2f}"
      "   <- silent failure")
print(f"shift check flags {flagged(X_src):.0%} of conventional and {flagged(X_des):.0%} "
      "of design candidates")
cov, inf = coverage(X_des, y_des)
print(f"design region, with shift check, no target data  not certified for {inf:.0%} of candidates")

layer.recalibrate(X_rec, y_rec)               # spend the target-measurement budget
print(f"design region, recalibrated on m = 30 records    {coverage(X_des, y_des)[0]:.2f}")

d = layer.decide(X_des, tau=6.0, direction="ge")          # specification: y >= 6
labels, counts = np.unique(np.asarray(d), return_counts=True)
print("decisions for design candidates:", dict(zip(labels.tolist(), counts.tolist())))
