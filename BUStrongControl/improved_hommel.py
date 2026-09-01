"""Last-step power enhancements of Hommel for CMS objective functions."""

from statistics import NormalDist
from typing import Iterable

import numpy as np

try:
    from scipy.special import ndtr, ndtri
except ModuleNotFoundError:  # pragma: no cover - SciPy is preferred for batches.
    _STD_NORMAL = NormalDist()

    def ndtri(values):
        values = np.asarray(values, dtype=float)
        return np.vectorize(_STD_NORMAL.inv_cdf, otypes=[float])(values)

    def ndtr(values):
        values = np.asarray(values, dtype=float)
        return np.vectorize(_STD_NORMAL.cdf, otypes=[float])(values)


def _validate_objective(objective: str) -> str:
    normalized = objective.lower().replace("_", "")
    if normalized in {"bayes", "mix", "pimix"}:
        return "bayes"
    if normalized in {"pi1", "1"}:
        return "pi1"
    raise ValueError("objective must be 'bayes'/'mix' or 'pi1'.")


def _hommel_rejections_sorted_batch(
    p_sorted: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Vectorized Hommel rejections for rows that are sorted increasingly."""
    p_sorted = np.asarray(p_sorted, dtype=float)
    if p_sorted.ndim != 2:
        raise ValueError("p_sorted must be a two-dimensional array.")
    if np.any(np.diff(p_sorted, axis=1) < 0.0):
        raise ValueError("Every row of p_sorted must be sorted increasingly.")

    n_tests = p_sorted.shape[1]
    adjusted = p_sorted.copy()
    for m in range(n_tests, 1, -1):
        scale = m / np.arange(1, m + 1, dtype=float)
        local_minimum = np.min(p_sorted[:, -m:] * scale, axis=1)
        adjusted[:, -m:] = np.maximum(
            adjusted[:, -m:],
            local_minimum[:, None],
        )
        if n_tests > m:
            adjusted[:, :-m] = np.maximum(
                adjusted[:, :-m],
                np.minimum(
                    m * p_sorted[:, :-m],
                    local_minimum[:, None],
                ),
            )

    adjusted = np.maximum.accumulate(adjusted, axis=1)
    return adjusted <= alpha


def improved_hommel_candidates_sorted_batch(
    p_sorted: np.ndarray,
    alpha: float = 0.05,
) -> np.ndarray:
    """Return the legacy leave-one-out Hommel candidate decisions.

    For sorted p-values, hypotheses 2..K use Hommel after removing p1. The
    first hypothesis uses its Hommel decision after removing p2, matching the
    published/legacy last-step construction.
    """
    p_sorted = np.asarray(p_sorted, dtype=float)
    if p_sorted.ndim != 2 or p_sorted.shape[1] < 2:
        raise ValueError("p_sorted must have shape (n_samples, K) with K >= 2.")
    if np.any((p_sorted < 0.0) | (p_sorted > 1.0)):
        raise ValueError("p-values must lie in [0, 1].")
    if np.any(np.diff(p_sorted, axis=1) < 0.0):
        raise ValueError("Every row of p_sorted must be sorted increasingly.")

    candidates = np.zeros_like(p_sorted, dtype=bool)
    without_first = p_sorted[:, 1:]
    candidates[:, 1:] = _hommel_rejections_sorted_batch(
        without_first,
        alpha,
    )

    without_second = np.concatenate(
        [p_sorted[:, :1], p_sorted[:, 2:]],
        axis=1,
    )
    candidates[:, 0] = _hommel_rejections_sorted_batch(
        without_second,
        alpha,
    )[:, 0]
    return candidates


def improved_hommel_scores_sorted_batch(
    p_sorted: np.ndarray,
    candidates: np.ndarray,
    theta: float,
    objective: str,
) -> np.ndarray:
    """Calculate improved-Hommel scores for sorted p-value rows."""
    objective = _validate_objective(objective)
    p_sorted = np.asarray(p_sorted, dtype=float)
    candidates = np.asarray(candidates, dtype=bool)
    if p_sorted.ndim != 2 or candidates.shape != p_sorted.shape:
        raise ValueError("p_sorted and candidates must have the same 2D shape.")

    clipped = np.clip(p_sorted, 1e-15, 1.0 - 1e-15)
    density = np.exp(theta * ndtri(clipped) - 0.5 * theta**2)
    if objective == "pi1":
        return np.sum(candidates * density, axis=1)

    K = p_sorted.shape[1]
    product_all = np.prod(1.0 + density, axis=1, keepdims=True)
    weights = density * product_all / (1.0 + density) / (2.0**K)
    return np.sum(candidates * weights, axis=1)


def improved_hommel_score(
    p_values: Iterable[float],
    theta: float,
    objective: str,
    alpha: float = 0.05,
) -> float:
    """Calculate one objective-specific improved-Hommel score."""
    p_values = np.asarray(p_values, dtype=float)
    if p_values.ndim != 1 or p_values.size < 2:
        raise ValueError("p_values must be a one-dimensional vector of length >= 2.")
    p_sorted = np.sort(p_values)[None, :]
    candidates = improved_hommel_candidates_sorted_batch(p_sorted, alpha)
    return float(
        improved_hommel_scores_sorted_batch(
            p_sorted,
            candidates,
            theta,
            objective,
        )[0]
    )


def apply_improved_hommel(
    p_values: Iterable[float],
    theta: float,
    threshold: float,
    objective: str,
    alpha: float = 0.05,
) -> dict:
    """Apply the last-step improved-Hommel procedure to one p-value vector."""
    p_values = np.asarray(p_values, dtype=float)
    if p_values.ndim != 1 or p_values.size < 2:
        raise ValueError("p_values must be a one-dimensional vector of length >= 2.")
    if np.any((p_values < 0.0) | (p_values > 1.0)):
        raise ValueError("p-values must lie in [0, 1].")

    order = np.argsort(p_values)
    p_sorted = p_values[order][None, :]
    candidates_sorted = improved_hommel_candidates_sorted_batch(
        p_sorted,
        alpha,
    )
    score = float(
        improved_hommel_scores_sorted_batch(
            p_sorted,
            candidates_sorted,
            theta,
            objective,
        )[0]
    )
    decisions_sorted = candidates_sorted[0].astype(int) * int(score > threshold)
    candidates = np.zeros_like(decisions_sorted)
    decisions = np.zeros_like(decisions_sorted)
    candidates[order] = candidates_sorted[0].astype(int)
    decisions[order] = decisions_sorted
    return {
        "decisions": decisions,
        "candidate_decisions": candidates,
        "score": score,
        "threshold": float(threshold),
    }


def _sample_null_pvalues(
    rng: np.random.Generator,
    n_samples: int,
    K: int,
    correlation: float,
) -> np.ndarray:
    if correlation == 0.0:
        return rng.random((n_samples, K))

    minimum = -1.0 / (K - 1)
    if not (minimum <= correlation < 1.0):
        raise ValueError(f"correlation must be in [{minimum}, 1).")
    covariance = np.full((K, K), correlation, dtype=float)
    np.fill_diagonal(covariance, 1.0)
    z_values = rng.multivariate_normal(
        mean=np.zeros(K),
        cov=covariance,
        size=n_samples,
    )
    return ndtr(z_values)


def estimate_improved_hommel_thresholds(
    K: int,
    B: int,
    alpha: float,
    target_powers: Iterable[float],
    seed: int = 123,
    chunk_size: int = 20_000,
    null_correlation: float = 0.0,
    quantile_method: str = "higher",
    verbose: bool = True,
) -> list[dict]:
    """Estimate improved-Hommel thresholds under the global null.

    The same Monte Carlo p-values and leave-one-out candidates are reused for
    all objective/target-power combinations.
    """
    if K < 2:
        raise ValueError("K must be at least 2.")
    if B <= 0 or chunk_size <= 0:
        raise ValueError("B and chunk_size must be positive.")

    target_powers = [float(value) for value in target_powers]
    normal = NormalDist()
    specifications = []
    for objective in ("bayes", "pi1"):
        for target_power in target_powers:
            if not (0.0 < target_power < 1.0):
                raise ValueError("Every target power must lie in (0, 1).")
            theta = normal.inv_cdf(alpha / K) - normal.inv_cdf(target_power)
            specifications.append(
                {
                    "name": (
                        f"Improved_Hommel_{objective}_tp_"
                        f"{str(target_power).replace('.', 'p')}"
                    ),
                    "objective": objective,
                    "target_power": target_power,
                    "theta": float(theta),
                    "scores": [],
                }
            )

    rng = np.random.default_rng(seed)
    completed = 0
    while completed < B:
        n_chunk = min(chunk_size, B - completed)
        p_sorted = np.sort(
            _sample_null_pvalues(
                rng,
                n_chunk,
                K,
                null_correlation,
            ),
            axis=1,
        )
        candidates = improved_hommel_candidates_sorted_batch(p_sorted, alpha)
        for specification in specifications:
            specification["scores"].append(
                improved_hommel_scores_sorted_batch(
                    p_sorted,
                    candidates,
                    specification["theta"],
                    specification["objective"],
                )
            )
        completed += n_chunk
        if verbose:
            print(f"Completed {completed}/{B} global-null samples")

    output = []
    for specification in specifications:
        scores = np.concatenate(specification.pop("scores"))
        threshold = float(
            np.quantile(scores, 1.0 - alpha, method=quantile_method)
        )
        tail_probability = float(np.mean(scores > threshold))
        output.append(
            {
                **specification,
                "threshold": threshold,
                "empirical_tail_probability": tail_probability,
            }
        )
        if verbose:
            print(
                f"{specification['name']}: threshold={threshold:.10g}, "
                f"P(score > threshold)={tail_probability:.6f}"
            )
    return output
