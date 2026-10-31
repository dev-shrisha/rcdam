"""Statistical helper utilities used throughout the RC‑DAM project."""
import numpy as np
from scipy.stats import binom, binomtest, norm

# ----------------------------------------------------------------------
# Wilson score interval (two‑sided)
# ----------------------------------------------------------------------
def wilson_interval(k, n, conf=0.95):
    """Wilson score interval for a proportion. Always report it next to k and n."""
    ci = binomtest(int(k), int(n)).proportion_ci(confidence_level=conf,
                                                  method="wilson")
    return float(ci.low), float(ci.high)


# ----------------------------------------------------------------------
# One‑sided Clopper‑Pearson upper bound (“rule of three” when k=0)
# ----------------------------------------------------------------------
def exact_upper_bound(k, n, conf=0.95):
    """One‑sided Clopper‑Pearson upper bound; with k = 0 this is the 'rule of three' bound."""
    ci = binomtest(int(k), int(n)).proportion_ci(confidence_level=1 - 2 * (1 - conf),
                                                  method="exact")
    return float(ci.high)


# ----------------------------------------------------------------------
# Exact two‑sided McNemar test
# ----------------------------------------------------------------------
def mcnemar_exact(b, c):
    """Exact two‑sided paired test. b = messages variant A missed and B delivered;
    c = the reverse."""
    n = b + c
    if n == 0:
        return 1.0
    return float(min(1.0, 2.0 * binom.cdf(min(b, c), n, 0.5)))


# ----------------------------------------------------------------------
# Zero‑miss sample size (rule‑of‑three style)
# ----------------------------------------------------------------------
def n_zero_miss(eps, conf=0.95):
    """Messages needed with ZERO misses to show miss rate < eps at the given confidence."""
    return int(np.ceil(np.log(1 - conf) / np.log(1 - eps)))


# ----------------------------------------------------------------------
# Sample size for detecting a difference between two proportions
# ----------------------------------------------------------------------
def n_for_difference(p1, p2, alpha=0.05, power=0.8):
    """Messages per variant to detect p1 versus p2 (normal approximation;
    unpaired, independent messages)."""
    z_a, z_b = norm.isf(alpha / 2), norm.isf(1 - power)
    pbar = 0.5 * (p1 + p2)
    num = (z_a * np.sqrt(2 * pbar * (1 - pbar)) +
           z_b * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return int(np.ceil(num / (p1 - p2) ** 2))


# ----------------------------------------------------------------------
# Cluster bootstrap for paired differences
# ----------------------------------------------------------------------
def cluster_bootstrap_diff(a, b, clusters, n_boot=4000, seed=0, conf=0.95):
    """
    Paired difference mean(a - b) with a cluster bootstrap.
    Use independent seeds (or weather events) as clusters, never single messages:
    messages inside one fade are strongly correlated.
    """
    a, b, clusters = np.asarray(a, float), np.asarray(b, float), np.asarray(clusters)
    ids = np.unique(clusters)
    per = np.array([(a[clusters == i] - b[clusters == i]).mean() for i in ids])
    rng = np.random.default_rng(seed)
    boots = np.array([per[rng.integers(0, len(per), len(per))].mean() for _ in range(n_boot)])
    lo, hi = np.quantile(boots, [(1 - conf) / 2, 1 - (1 - conf) / 2])
    return float(per.mean()), float(lo), float(hi)


# ----------------------------------------------------------------------
# Matched‑reliability sweep: cheapest config whose Wilson upper bound ≤ target
# ----------------------------------------------------------------------
def overhead_at_target(miss_k, miss_n, overhead, target, conf=0.95):
    """
    Matched‑reliability comparison: the cheapest configuration of a sweep whose
    miss‑rate UPPER bound is below target. Returns the index, or None if no
    configuration is certified.
    """
    best = None
    for i, (k, n, o) in enumerate(zip(miss_k, miss_n, overhead)):
        if wilson_interval(k, n, conf)[1] <= target and (best is None or o < overhead[best]):
            best = i
    return best
