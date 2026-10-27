"""Uncertainty for experiment values (DESIGN 21): 95% intervals, recorded next to the value.

A rate gets a Wilson interval; a mean or a difference of means gets a percentile bootstrap over
the items it averages, with a fixed seed so the interval is as reproducible as the value."""
from __future__ import annotations

import math

import numpy as np

Z95 = 1.959964


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float] | None:
    """Interval for a rate k/n; sensible at 0/n and n/n, unlike the normal approximation."""
    if n == 0:
        return None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return round(max(0.0, centre - half), 6), round(min(1.0, centre + half), 6)


def bootstrap(statistic, *samples, n_boot: int = 2000, seed: int = 0) -> tuple[float, float] | None:
    """Percentile interval for statistic(*samples), resampling each sample's items with replacement.
    Paired samples (the same items before and after an intervention) are passed as one array of
    pairs, so they are resampled together."""
    arrays = [np.asarray(s, dtype=float) for s in samples]
    if any(len(a) < 2 for a in arrays):
        return None
    rng = np.random.default_rng(seed)
    stats = [statistic(*(a[rng.integers(0, len(a), len(a))] for a in arrays)) for _ in range(n_boot)]
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return round(float(lo), 6), round(float(hi), 6)
