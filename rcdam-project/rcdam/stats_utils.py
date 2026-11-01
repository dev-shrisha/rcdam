"""Statistical helper utilities used throughout the RC-DAM project.

Confidence convention:
- wilson_interval(..., conf=0.95) returns a TWO-SIDED 95% interval.
- wilson_lower_bound(..., conf=0.95) returns a ONE-SIDED 95% lower bound.
- exact_upper_bound(..., conf=0.95) returns a ONE-SIDED 95% Clopper-Pearson upper bound.
"""
import numpy as np
from scipy.stats import beta, binom, binomtest, norm


def _validate_confidence(conf):
    """Validate a confidence level strictly between zero and one."""
    if not np.isfinite(conf) or not 0.0 < conf < 1.0:
        raise ValueError("conf must be strictly between 0 and 1")


def _validate_binomial_counts(k, n):
    """Validate and normalize binomial success/failure counts."""
    if isinstance(k, (bool, np.bool_)) or isinstance(n, (bool, np.bool_)):
        raise ValueError("k and n must be integer counts, not booleans")
    if not np.isfinite(k) or not np.isfinite(n):
        raise ValueError("k and n must be finite")
    if int(k) != k or int(n) != n:
        raise ValueError("k and n must be integer counts")
    k, n = int(k), int(n)
    if n <= 0:
        raise ValueError("n must be greater than zero")
    if not 0 <= k <= n:
        raise ValueError("k must satisfy 0 <= k <= n")
    return k, n


# ----------------------------------------------------------------------
# Wilson score interval (two-sided)
# ----------------------------------------------------------------------
def wilson_interval(k, n, conf=0.95):
    """Return the two-sided Wilson score interval for a binomial proportion."""
    _validate_confidence(conf)
    k, n = _validate_binomial_counts(k, n)
    ci = binomtest(k, n).proportion_ci(
        confidence_level=conf,
        method="wilson",
    )
    return float(ci.low), float(ci.high)


# ----------------------------------------------------------------------
# Wilson score lower bound (one-sided)
# ----------------------------------------------------------------------
def wilson_lower_bound(k, n, conf=0.95):
    """Return a one-sided Wilson lower confidence bound for a proportion.

    Use this when the decision rule is specifically:
        lower confidence bound >= required success probability.

    This is not the same as taking the lower endpoint of a two-sided
    interval with the same numerical confidence level.
    """
    _validate_confidence(conf)
    k, n = _validate_binomial_counts(k, n)

    p_hat = k / n
    z = float(norm.ppf(conf))
    z2 = z * z
    denominator = 1.0 + z2 / n
    center = p_hat + z2 / (2.0 * n)
    margin = z * np.sqrt(
        p_hat * (1.0 - p_hat) / n + z2 / (4.0 * n * n)
    )
    lower = (center - margin) / denominator
    return float(max(0.0, lower))


# ----------------------------------------------------------------------
# One-sided Clopper-Pearson upper bound ("rule of three" when k=0)
# ----------------------------------------------------------------------
def exact_upper_bound(k, n, conf=0.95):
    """Return a one-sided exact Clopper-Pearson upper bound for a rate."""
    _validate_confidence(conf)
    k, n = _validate_binomial_counts(k, n)
    if k == n:
        return 1.0
    # Beta quantile gives the one-sided upper endpoint at confidence conf.
    return float(beta.ppf(conf, k + 1, n - k))


# ----------------------------------------------------------------------
# Exact two-sided McNemar test
# ----------------------------------------------------------------------
def mcnemar_exact(b, c):
    """Return the exact two-sided McNemar p-value.

    b: pairs where variant A missed and variant B delivered.
    c: pairs where variant A delivered and variant B missed.
    """
    if isinstance(b, (bool, np.bool_)) or isinstance(c, (bool, np.bool_)):
        raise ValueError("b and c must be non-negative integer counts")
    if not np.isfinite(b) or not np.isfinite(c) or int(b) != b or int(c) != c:
        raise ValueError("b and c must be non-negative integer counts")
    b, c = int(b), int(c)
    if b < 0 or c < 0:
        raise ValueError("b and c must be non-negative")
    n_discordant = b + c
    if n_discordant == 0:
        return 1.0
    return float(min(1.0, 2.0 * binom.cdf(min(b, c), n_discordant, 0.5)))


# ----------------------------------------------------------------------
# Zero-miss sample size (exact one-sided confidence bound)
# ----------------------------------------------------------------------
def n_zero_miss(eps, conf=0.95):
    """Minimum n with zero misses so the exact upper rate bound is <= eps."""
    _validate_confidence(conf)
    if not np.isfinite(eps) or not 0.0 < eps < 1.0:
        raise ValueError("eps must be strictly between 0 and 1")
    return int(np.ceil(np.log1p(-conf) / np.log1p(-eps)))


# ----------------------------------------------------------------------
# Sample size for detecting a difference between two proportions
# ----------------------------------------------------------------------
def n_for_difference(p1, p2, alpha=0.05, power=0.8):
    """Approximate samples per group for two independent proportions."""
    for name, value in (("p1", p1), ("p2", p2)):
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    if p1 == p2:
        raise ValueError("p1 and p2 must differ")
    if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")
    if not np.isfinite(power) or not 0.0 < power < 1.0:
        raise ValueError("power must be strictly between 0 and 1")

    z_alpha = norm.isf(alpha / 2.0)
    z_power = norm.ppf(power)
    p_bar = 0.5 * (p1 + p2)
    numerator = (
        z_alpha * np.sqrt(2.0 * p_bar * (1.0 - p_bar))
        + z_power * np.sqrt(p1 * (1.0 - p1) + p2 * (1.0 - p2))
    ) ** 2
    return int(np.ceil(numerator / (p1 - p2) ** 2))


# ----------------------------------------------------------------------
# Cluster bootstrap for paired differences
# ----------------------------------------------------------------------
def cluster_bootstrap_diff(a, b, clusters, n_boot=4000, seed=0, conf=0.95):
    """Return mean paired difference and a cluster-bootstrap confidence interval.

    Each cluster (for example, an independent seed or weather event) is
    resampled as a unit because messages within a fade event are correlated.
    Cluster means receive equal weight.
    """
    _validate_confidence(conf)
    if isinstance(n_boot, (bool, np.bool_)) or int(n_boot) != n_boot or n_boot <= 0:
        raise ValueError("n_boot must be a positive integer")

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    clusters = np.asarray(clusters)

    if a.ndim != 1 or b.ndim != 1 or clusters.ndim != 1:
        raise ValueError("a, b, and clusters must be one-dimensional")
    if not (len(a) == len(b) == len(clusters)):
        raise ValueError("a, b, and clusters must have equal lengths")
    if len(a) == 0:
        raise ValueError("input arrays must not be empty")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("a and b must contain only finite values")

    ids = np.unique(clusters)
    if len(ids) < 2:
        raise ValueError("at least two independent clusters are required")

    per_cluster = np.array([
        (a[clusters == cluster] - b[clusters == cluster]).mean()
        for cluster in ids
    ])
    rng = np.random.default_rng(seed)
    boot_means = np.empty(int(n_boot), dtype=float)
    for i in range(int(n_boot)):
        sampled = rng.integers(0, len(per_cluster), size=len(per_cluster))
        boot_means[i] = per_cluster[sampled].mean()

    tail = (1.0 - conf) / 2.0
    lo, hi = np.quantile(boot_means, [tail, 1.0 - tail])
    return float(per_cluster.mean()), float(lo), float(hi)


# ----------------------------------------------------------------------
# Matched-reliability sweep: cheapest configuration with certified miss rate
# ----------------------------------------------------------------------
def overhead_at_target(miss_k, miss_n, overhead, target, conf=0.95):
    """Return the cheapest index whose Wilson upper miss-rate bound <= target.

    Returns None if no configuration in the sweep is certified.
    """
    _validate_confidence(conf)
    if not np.isfinite(target) or not 0.0 < target < 1.0:
        raise ValueError("target must be strictly between 0 and 1")
    if not (len(miss_k) == len(miss_n) == len(overhead)):
        raise ValueError("miss_k, miss_n, and overhead must have equal lengths")

    best = None
    for i, (k, n, cost) in enumerate(zip(miss_k, miss_n, overhead)):
        if not np.isfinite(cost):
            continue
        _, upper = wilson_interval(k, n, conf)
        if upper <= target and (best is None or cost < overhead[best]):
            best = i
    return best
