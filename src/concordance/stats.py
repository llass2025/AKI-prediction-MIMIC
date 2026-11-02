"""Wilson intervals, Cohen's kappa with a cell-multinomial bootstrap, small-cell suppression."""

import math
import zlib

import numpy as np

Z95 = 1.959963984540054
SUPPRESSED = "suppressed"


def wilson(k: float, n: float, z: float = Z95):
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def kappa_arr(tp, fp, fn, tn):
    tp, fp, fn, tn = (np.asarray(x, dtype=float) for x in (tp, fp, fn, tn))
    n = tp + fp + fn + tn
    with np.errstate(divide="ignore", invalid="ignore"):
        po = (tp + tn) / n
        pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n)
        k = (po - pe) / (1.0 - pe)
    return np.where((n > 0) & (pe < 1.0), k, np.nan)


def kappa(tp, fp, fn, tn) -> float:
    return float(kappa_arr(tp, fp, fn, tn))


def bootstrap_kappa_ci(cells, n_boot: int, seed: int, key: str):
    """
    Percentile CI for kappa. Resampling admissions with replacement is
    equivalent to a multinomial draw over the four cell counts, which is what
    we do (fast, and deterministic: the stream depends only on seed + key).
    """
    n = int(sum(cells))
    if n == 0 or n_boot <= 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(np.random.SeedSequence([seed, zlib.crc32(key.encode())]))
    draws = rng.multinomial(n, np.asarray(cells, dtype=float) / n, size=n_boot)
    k = kappa_arr(draws[:, 0], draws[:, 1], draws[:, 2], draws[:, 3])
    k = k[~np.isnan(k)]
    if k.size == 0:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(k, [2.5, 97.5])
    return (float(lo), float(hi))


class Suppressor:
    """Counts below min_cell print as '<min_cell'; rates built from such cells print 'suppressed'."""

    def __init__(self, min_cell: int):
        self.min_cell = int(min_cell)

    def ok(self, *cells) -> bool:
        return all(c >= self.min_cell for c in cells)

    def count(self, n) -> str:
        return str(int(n)) if n >= self.min_cell else f"<{self.min_cell}"

    def value(self, x: float, nd: int, *cells) -> str:
        if not self.ok(*cells):
            return SUPPRESSED
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return "NA"
        return f"{x:.{nd}f}"

    def pct(self, num, den, *cells) -> str:
        cells = cells or (num, den)
        return self.value(100.0 * num / den if den else float("nan"), 2, *cells)
