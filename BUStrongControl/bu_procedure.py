from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from itertools import combinations, product
from typing import Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests


Subset = Tuple[int, ...]
ThresholdMap = Dict[Subset, float]
LocalTest = Callable[[Sequence[float]], int]
LocalTestMap = Dict[Subset, LocalTest]
ScoreFunction = Callable[[Sequence[float]], float]
ScoreFunctionMap = Dict[Subset, ScoreFunction]

_P_CLIP_EPS = 1e-12


def all_subsets_of_size(k_total: int, subset_size: int) -> List[Subset]:
    return [tuple(subset) for subset in combinations(range(k_total), subset_size)]


def proper_subsets_containing_i(I: Subset, i: int) -> List[Subset]:
    subsets: List[Subset] = []
    for subset_size in range(1, len(I)):
        for subset in combinations(I, subset_size):
            if i in subset:
                subsets.append(tuple(subset))
    return subsets


def principal_submatrix(Sigma: np.ndarray, I: Subset) -> np.ndarray:
    idx = np.array(I, dtype=int)
    return Sigma[np.ix_(idx, idx)]


def _clip_pvalues(p_I: Sequence[float]) -> np.ndarray:
    return np.clip(np.asarray(p_I, dtype=float), _P_CLIP_EPS, 1.0 - _P_CLIP_EPS)


def _subset_local_index(I: Subset) -> Dict[int, int]:
    return {global_idx: local_idx for local_idx, global_idx in enumerate(I)}


def _binary_vectors_for_size(m: int) -> np.ndarray:
    return np.asarray(list(product((0, 1), repeat=m)), dtype=float)


def all_binary_configurations(K: int) -> List[Tuple[int, ...]]:
    return [tuple(int(bit) for bit in h) for h in product((0, 1), repeat=K)]


def _pvalues_from_z(z_I: np.ndarray, Sigma_II: np.ndarray) -> np.ndarray:
    std = np.sqrt(np.diag(Sigma_II))
    return 1.0 - norm.cdf(z_I / std)


def _validate_configuration(h: Sequence[int], K: int) -> np.ndarray:
    h_vec = np.asarray(h, dtype=int)
    if h_vec.shape != (K,):
        raise ValueError("Configuration h must have shape (K,).")
    if not np.all(np.isin(h_vec, [0, 1])):
        raise ValueError("Configuration h must contain only 0/1 entries.")
    return h_vec


def subset_coefficients_pvalue(
    I: Subset,
    p_I: Sequence[float],
    Sigma: np.ndarray,
    theta_alt: float,
) -> Dict[int, float]:
    """Return null-density-normalized coefficients for the subset score.

    Each coefficient is a prior-weighted sum of Gaussian likelihood ratios
    over configurations in which the corresponding hypothesis is non-null:

        2**(-|I|) * sum_{h: h_i=1} f_h(z_I) / f_0(z_I).

    Computing the ratio in z-space cancels the p-value transformation
    Jacobian and is more stable than dividing p-value densities directly.
    """
    Sigma_II = principal_submatrix(Sigma, I)
    p_vec = _clip_pvalues(p_I)
    local_index = _subset_local_index(I)
    std = np.sqrt(np.diag(Sigma_II))
    q = norm.ppf(1.0 - p_vec)
    z_I = std * q

    m = len(I)
    h_vectors = _binary_vectors_for_size(m)
    mean_vectors = theta_alt * h_vectors
    log_weights = np.full(h_vectors.shape[0], -m * np.log(2.0))

    # For equal-covariance Gaussians,
    # log(f_h(z) / f_0(z)) = mu_h' Sigma^{-1} z
    #                         - 0.5 mu_h' Sigma^{-1} mu_h.
    precision_times_z = np.linalg.solve(Sigma_II, z_I)
    precision_times_means = np.linalg.solve(Sigma_II, mean_vectors.T).T
    log_likelihood_ratios = (
        mean_vectors @ precision_times_z
        - 0.5 * np.einsum("ij,ij->i", mean_vectors, precision_times_means)
    )

    coefficients: Dict[int, float] = {}
    for global_idx in I:
        r = local_index[global_idx]
        mask = h_vectors[:, r] == 1.0
        log_coefficient = logsumexp(
            log_weights[mask] + log_likelihood_ratios[mask]
        )
        coefficients[global_idx] = float(np.exp(log_coefficient))

    return coefficients


def subset_score(
    I: Subset,
    p_I: Sequence[float],
    local_test: Mapping[Subset, LocalTest],
    Sigma: np.ndarray,
    theta_alt: float,
) -> float:
    p_vec = _clip_pvalues(p_I)
    coefficients = subset_coefficients_pvalue(I, p_vec, Sigma, theta_alt)
    local_index = _subset_local_index(I)

    score = 0.0
    for i in I:
        prod_value = 1
        for J in proper_subsets_containing_i(I, i):
            p_J = np.asarray([p_vec[local_index[j]] for j in J], dtype=float)
            prod_value *= int(local_test[J](p_J))
            if prod_value == 0:
                break
        score += coefficients[i] * prod_value

    return float(score)


def _subset_score_from_thresholds(
    I: Subset,
    p_I: Sequence[float],
    threshold: Mapping[Subset, float],
    Sigma: np.ndarray,
    theta_alt: float,
    proper_subset_cache: Mapping[Tuple[Subset, int], List[Subset]] | None = None,
) -> float:
    p_vec = _clip_pvalues(p_I)
    coefficients = subset_coefficients_pvalue(I, p_vec, Sigma, theta_alt)
    local_index = _subset_local_index(I)

    score = 0.0
    for i in I:
        prod_value = 1
        subsets = (
            proper_subset_cache[(I, i)]
            if proper_subset_cache is not None and (I, i) in proper_subset_cache
            else proper_subsets_containing_i(I, i)
        )
        for J in subsets:
            p_J = np.asarray([p_vec[local_index[j]] for j in J], dtype=float)
            prod_value *= _evaluate_local_test_from_thresholds(
                I=J,
                p_I=p_J,
                threshold=threshold,
                Sigma=Sigma,
                theta_alt=theta_alt,
                proper_subset_cache=proper_subset_cache,
            )
            if prod_value == 0:
                break
        score += coefficients[i] * prod_value

    return float(score)


def _evaluate_local_test_from_thresholds(
    I: Subset,
    p_I: Sequence[float],
    threshold: Mapping[Subset, float],
    Sigma: np.ndarray,
    theta_alt: float,
    proper_subset_cache: Mapping[Tuple[Subset, int], List[Subset]] | None = None,
) -> int:
    p_vec = _clip_pvalues(p_I)
    if len(I) == 1:
        return int(p_vec[0] <= threshold[I])
    return int(
        _subset_score_from_thresholds(
            I=I,
            p_I=p_vec,
            threshold=threshold,
            Sigma=Sigma,
            theta_alt=theta_alt,
            proper_subset_cache=proper_subset_cache,
        )
        > threshold[I]
    )


def _score_function_dispatch(
    p_I: Sequence[float],
    *,
    subset: Subset,
    threshold: Mapping[Subset, float],
    Sigma: np.ndarray,
    theta_alt: float,
    proper_subset_cache: Mapping[Tuple[Subset, int], List[Subset]],
) -> float:
    return _subset_score_from_thresholds(
        I=subset,
        p_I=p_I,
        threshold=threshold,
        Sigma=Sigma,
        theta_alt=theta_alt,
        proper_subset_cache=proper_subset_cache,
    )


def _local_test_dispatch(
    p_I: Sequence[float],
    *,
    subset: Subset,
    threshold: Mapping[Subset, float],
    Sigma: np.ndarray,
    theta_alt: float,
    proper_subset_cache: Mapping[Tuple[Subset, int], List[Subset]],
) -> int:
    return _evaluate_local_test_from_thresholds(
        I=subset,
        p_I=p_I,
        threshold=threshold,
        Sigma=Sigma,
        theta_alt=theta_alt,
        proper_subset_cache=proper_subset_cache,
    )


def calibrate_threshold_for_subset(
    I: Subset,
    local_test: Mapping[Subset, LocalTest],
    Sigma: np.ndarray,
    alpha: float,
    B: int,
    theta_alt: float,
    rng: np.random.Generator,
) -> float:
    Sigma_II = principal_submatrix(Sigma, I)
    scores = np.empty(B, dtype=float)

    for b in range(B):
        z_I = rng.multivariate_normal(mean=np.zeros(len(I), dtype=float), cov=Sigma_II)
        p_I = _clip_pvalues(_pvalues_from_z(z_I, Sigma_II))
        scores[b] = subset_score(I, p_I, local_test, Sigma, theta_alt)

    scores_sorted = np.sort(scores)
    idx = int(np.ceil((1.0 - alpha) * (B + 1))) - 1
    idx = min(max(idx, 0), B - 1)
    return float(scores_sorted[idx])


def _calibrate_subset_worker(
    subset: Subset,
    threshold: Mapping[Subset, float],
    Sigma: np.ndarray,
    alpha: float,
    B: int,
    theta_alt: float,
    random_seed: int,
    proper_subset_cache: Mapping[Tuple[Subset, int], List[Subset]],
) -> Tuple[Subset, float]:
    rng = np.random.default_rng(random_seed)
    Sigma_II = principal_submatrix(Sigma, subset)
    scores = np.empty(B, dtype=float)

    for b in range(B):
        z_I = rng.multivariate_normal(mean=np.zeros(len(subset), dtype=float), cov=Sigma_II)
        p_I = _clip_pvalues(_pvalues_from_z(z_I, Sigma_II))
        scores[b] = _subset_score_from_thresholds(
            I=subset,
            p_I=p_I,
            threshold=threshold,
            Sigma=Sigma,
            theta_alt=theta_alt,
            proper_subset_cache=proper_subset_cache,
        )

    scores_sorted = np.sort(scores)
    idx = int(np.ceil((1.0 - alpha) * (B + 1))) - 1
    idx = min(max(idx, 0), B - 1)
    return subset, float(scores_sorted[idx])


def _fit_payload_from_object(fit_object: BUProcedureFit) -> Dict[str, object]:
    return {
        "Sigma": np.asarray(fit_object.Sigma, dtype=float),
        "theta_alt": float(fit_object.theta_alt),
        "alpha": float(fit_object.alpha),
        "threshold": dict(fit_object.threshold),
        "subsets_by_size": dict(fit_object.subsets_by_size),
        "proper_subset_cache": dict(fit_object.proper_subset_cache),
        "num_workers": int(fit_object.num_workers),
    }


def _fit_object_from_payload(payload: Mapping[str, object]) -> BUProcedureFit:
    score_function, local_test = _build_callable_maps(
        threshold=payload["threshold"],
        Sigma=payload["Sigma"],
        theta_alt=payload["theta_alt"],
        subsets_by_size=payload["subsets_by_size"],
        proper_subset_cache=payload["proper_subset_cache"],
    )
    return BUProcedureFit(
        Sigma=np.asarray(payload["Sigma"], dtype=float),
        theta_alt=float(payload["theta_alt"]),
        alpha=float(payload["alpha"]),
        threshold=dict(payload["threshold"]),
        score_function=score_function,
        local_test=local_test,
        subsets_by_size=dict(payload["subsets_by_size"]),
        proper_subset_cache=dict(payload["proper_subset_cache"]),
        num_workers=int(payload["num_workers"]),
    )


def _simulation_chunk_worker(
    fit_payload: Mapping[str, object],
    n_sim: int,
    random_seed: int,
    h: np.ndarray | None,
    pi1: float,
) -> Dict[str, np.ndarray | float]:
    fit_object = _fit_object_from_payload(fit_payload)
    rng = np.random.default_rng(random_seed)
    K = fit_object.Sigma.shape[0]

    h_samples = np.empty((n_sim, K), dtype=int)
    z_samples = np.empty((n_sim, K), dtype=float)
    p_samples = np.empty((n_sim, K), dtype=float)
    decisions = np.empty((n_sim, K), dtype=int)
    any_rejection = np.empty(n_sim, dtype=int)
    false_rejection = np.empty(n_sim, dtype=int)
    true_rejection = np.empty(n_sim, dtype=int)
    fdp = np.zeros(n_sim, dtype=float)
    tdp = np.zeros(n_sim, dtype=float)

    for sim_idx in range(n_sim):
        h_vec, z_vec, p_vec = sample_gaussian_two_group(
            Sigma=fit_object.Sigma,
            theta_alt=fit_object.theta_alt,
            rng=rng,
            h=h,
            pi1=pi1,
        )
        decision_vec = apply_bu_procedure(p_vec, fit_object)
        null_mask = h_vec == 0
        alt_mask = h_vec == 1
        num_rejections = int(decision_vec.sum())
        num_false = int(np.sum(decision_vec[null_mask])) if np.any(null_mask) else 0
        num_true = int(np.sum(decision_vec[alt_mask])) if np.any(alt_mask) else 0

        h_samples[sim_idx] = h_vec
        z_samples[sim_idx] = z_vec
        p_samples[sim_idx] = p_vec
        decisions[sim_idx] = decision_vec
        any_rejection[sim_idx] = int(num_rejections > 0)
        false_rejection[sim_idx] = int(num_false > 0)
        true_rejection[sim_idx] = int(num_true > 0)
        fdp[sim_idx] = num_false / num_rejections if num_rejections > 0 else 0.0
        if np.any(alt_mask):
            tdp[sim_idx] = num_true / int(np.sum(alt_mask))

    return {
        "h_samples": h_samples,
        "z_samples": z_samples,
        "p_samples": p_samples,
        "decisions": decisions,
        "any_rejection": any_rejection,
        "false_rejection": false_rejection,
        "true_rejection": true_rejection,
        "fdp": fdp,
        "tdp": tdp,
    }


def _competition_chunk_worker(
    fit_payload: Mapping[str, object],
    n_sim: int,
    random_seed: int,
    h: np.ndarray | None,
    pi1: float,
) -> Dict[str, np.ndarray]:
    fit_object = _fit_object_from_payload(fit_payload)
    rng = np.random.default_rng(random_seed)
    K = fit_object.Sigma.shape[0]

    h_samples = np.empty((n_sim, K), dtype=int)
    z_samples = np.empty((n_sim, K), dtype=float)
    p_samples = np.empty((n_sim, K), dtype=float)
    bu_decisions = np.empty((n_sim, K), dtype=int)
    hommel_decisions = np.empty((n_sim, K), dtype=int)

    for sim_idx in range(n_sim):
        h_vec, z_vec, p_vec = sample_gaussian_two_group(
            Sigma=fit_object.Sigma,
            theta_alt=fit_object.theta_alt,
            rng=rng,
            h=h,
            pi1=pi1,
        )
        h_samples[sim_idx] = h_vec
        z_samples[sim_idx] = z_vec
        p_samples[sim_idx] = p_vec
        bu_decisions[sim_idx] = apply_bu_procedure(p_vec, fit_object)
        hommel_decisions[sim_idx] = apply_hommel_procedure(p_vec, alpha=fit_object.alpha)

    return {
        "h_samples": h_samples,
        "z_samples": z_samples,
        "p_samples": p_samples,
        "bu_decisions": bu_decisions,
        "hommel_decisions": hommel_decisions,
    }


@dataclass
class BUProcedureFit:
    Sigma: np.ndarray
    theta_alt: float
    alpha: float
    threshold: ThresholdMap
    score_function: ScoreFunctionMap
    local_test: LocalTestMap
    subsets_by_size: Dict[int, List[Subset]] = field(default_factory=dict)
    proper_subset_cache: Dict[Tuple[Subset, int], List[Subset]] = field(default_factory=dict)
    num_workers: int = 1

    def __getstate__(self) -> Dict[str, object]:
        state = dict(self.__dict__)
        state["score_function"] = {}
        state["local_test"] = {}
        return state

    def __setstate__(self, state: Mapping[str, object]) -> None:
        self.__dict__.update(state)
        rebuilt = _build_callable_maps(
            threshold=self.threshold,
            Sigma=self.Sigma,
            theta_alt=self.theta_alt,
            subsets_by_size=self.subsets_by_size,
            proper_subset_cache=self.proper_subset_cache,
        )
        self.score_function = rebuilt[0]
        self.local_test = rebuilt[1]


def _build_callable_maps(
    threshold: Mapping[Subset, float],
    Sigma: np.ndarray,
    theta_alt: float,
    subsets_by_size: Mapping[int, List[Subset]],
    proper_subset_cache: Mapping[Tuple[Subset, int], List[Subset]],
) -> Tuple[ScoreFunctionMap, LocalTestMap]:
    score_function: ScoreFunctionMap = {}
    local_test: LocalTestMap = {}
    max_size = max(subsets_by_size) if subsets_by_size else 0
    for subset_size in range(1, max_size + 1):
        for subset in subsets_by_size[subset_size]:
            if len(subset) >= 2:
                score_function[subset] = partial(
                    _score_function_dispatch,
                    subset=subset,
                    threshold=threshold,
                    Sigma=Sigma,
                    theta_alt=theta_alt,
                    proper_subset_cache=proper_subset_cache,
                )
            local_test[subset] = partial(
                _local_test_dispatch,
                subset=subset,
                threshold=threshold,
                Sigma=Sigma,
                theta_alt=theta_alt,
                proper_subset_cache=proper_subset_cache,
            )
    return score_function, local_test


def sample_gaussian_two_group(
    Sigma: np.ndarray,
    theta_alt: float,
    rng: np.random.Generator,
    h: Sequence[int] | None = None,
    pi1: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Sigma = np.asarray(Sigma, dtype=float)
    K = Sigma.shape[0]
    if h is None:
        if pi1 < 0.0 or pi1 > 1.0:
            raise ValueError("pi1 must lie in [0, 1].")
        h_vec = rng.binomial(1, pi1, size=K).astype(int)
    else:
        h_vec = _validate_configuration(h, K)

    mean = theta_alt * h_vec
    z = rng.multivariate_normal(mean=mean, cov=Sigma)
    p = _clip_pvalues(_pvalues_from_z(z, Sigma))
    return h_vec, z, p


def _resolve_num_workers(num_workers: int | None) -> int:
    if num_workers is None:
        return 1
    if num_workers <= 0:
        raise ValueError("num_workers must be a positive integer or None.")
    return int(num_workers)


def _split_evenly(total: int, num_parts: int) -> List[int]:
    base = total // num_parts
    remainder = total % num_parts
    return [base + (1 if part < remainder else 0) for part in range(num_parts)]


def _run_subset_calibration_jobs(
    subsets: Sequence[Subset],
    threshold: Mapping[Subset, float],
    Sigma: np.ndarray,
    alpha: float,
    B: int,
    theta_alt: float,
    seeds: Sequence[int],
    proper_subset_cache: Mapping[Tuple[Subset, int], List[Subset]],
    num_workers: int,
) -> List[Tuple[Subset, float]]:
    if num_workers == 1 or len(subsets) == 1:
        return [
            _calibrate_subset_worker(
                subset=I,
                threshold=threshold,
                Sigma=Sigma,
                alpha=alpha,
                B=B,
                theta_alt=theta_alt,
                random_seed=int(seed),
                proper_subset_cache=proper_subset_cache,
            )
            for I, seed in zip(subsets, seeds)
        ]

    try:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    _calibrate_subset_worker,
                    I,
                    threshold,
                    Sigma,
                    alpha,
                    B,
                    theta_alt,
                    int(seed),
                    proper_subset_cache,
                )
                for I, seed in zip(subsets, seeds)
            ]
            return [future.result() for future in futures]
    except (PermissionError, OSError):
        return [
            _calibrate_subset_worker(
                subset=I,
                threshold=threshold,
                Sigma=Sigma,
                alpha=alpha,
                B=B,
                theta_alt=theta_alt,
                random_seed=int(seed),
                proper_subset_cache=proper_subset_cache,
            )
            for I, seed in zip(subsets, seeds)
        ]


def _run_simulation_jobs(
    fit_payload: Mapping[str, object],
    chunk_sizes: Sequence[int],
    seeds: Sequence[int],
    h: np.ndarray | None,
    pi1: float,
    num_workers: int,
) -> List[Dict[str, np.ndarray | float]]:
    if num_workers == 1 or len(chunk_sizes) == 1:
        return [
            _simulation_chunk_worker(
                fit_payload=fit_payload,
                n_sim=chunk_size,
                random_seed=int(seed),
                h=h,
                pi1=pi1,
            )
            for chunk_size, seed in zip(chunk_sizes, seeds)
        ]

    try:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    _simulation_chunk_worker,
                    fit_payload,
                    chunk_size,
                    int(seed),
                    h,
                    pi1,
                )
                for chunk_size, seed in zip(chunk_sizes, seeds)
            ]
            return [future.result() for future in futures]
    except (PermissionError, OSError):
        return [
            _simulation_chunk_worker(
                fit_payload=fit_payload,
                n_sim=chunk_size,
                random_seed=int(seed),
                h=h,
                pi1=pi1,
            )
            for chunk_size, seed in zip(chunk_sizes, seeds)
        ]


def _run_competition_jobs(
    fit_payload: Mapping[str, object],
    chunk_sizes: Sequence[int],
    seeds: Sequence[int],
    h: np.ndarray | None,
    pi1: float,
    num_workers: int,
) -> List[Dict[str, np.ndarray]]:
    if num_workers == 1 or len(chunk_sizes) == 1:
        return [
            _competition_chunk_worker(
                fit_payload=fit_payload,
                n_sim=chunk_size,
                random_seed=int(seed),
                h=h,
                pi1=pi1,
            )
            for chunk_size, seed in zip(chunk_sizes, seeds)
        ]

    try:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    _competition_chunk_worker,
                    fit_payload,
                    chunk_size,
                    int(seed),
                    h,
                    pi1,
                )
                for chunk_size, seed in zip(chunk_sizes, seeds)
            ]
            return [future.result() for future in futures]
    except (PermissionError, OSError):
        return [
            _competition_chunk_worker(
                fit_payload=fit_payload,
                n_sim=chunk_size,
                random_seed=int(seed),
                h=h,
                pi1=pi1,
            )
            for chunk_size, seed in zip(chunk_sizes, seeds)
        ]


def fit_bu_procedure(
    Sigma: np.ndarray,
    theta_alt: float,
    alpha: float = 0.05,
    B: int = 10000,
    tail: str = "one_sided_upper",
    use_sorted_symmetry: bool = False,
    random_seed: int | None = None,
    n_jobs: int | None = 1,
    num_workers: int | None = None,
) -> BUProcedureFit:
    Sigma = np.asarray(Sigma, dtype=float)
    if Sigma.ndim != 2 or Sigma.shape[0] != Sigma.shape[1]:
        raise ValueError("Sigma must be a square covariance matrix.")
    if not np.allclose(Sigma, Sigma.T):
        raise ValueError("Sigma must be symmetric.")
    if alpha <= 0.0 or alpha >= 1.0:
        raise ValueError("alpha must lie in (0, 1).")
    if B <= 0:
        raise ValueError("B must be a positive integer.")
    if tail != "one_sided_upper":
        raise NotImplementedError("Only 'one_sided_upper' is currently supported.")
    if use_sorted_symmetry:
        raise NotImplementedError("use_sorted_symmetry is not implemented in this version.")

    K = Sigma.shape[0]
    if num_workers is not None and n_jobs != 1 and n_jobs != num_workers:
        raise ValueError("Pass only one of n_jobs or num_workers, or use matching values.")
    worker_count_requested = num_workers if num_workers is not None else n_jobs
    num_workers = _resolve_num_workers(worker_count_requested)
    rng = np.random.default_rng(random_seed)
    threshold: ThresholdMap = {}
    subsets_by_size: Dict[int, List[Subset]] = {
        subset_size: all_subsets_of_size(K, subset_size)
        for subset_size in range(1, K + 1)
    }
    proper_subset_cache: Dict[Tuple[Subset, int], List[Subset]] = {}

    for i in range(K):
        I = (i,)
        threshold[I] = float(alpha)

    for subset_size in range(2, K + 1):
        current_subsets = subsets_by_size[subset_size]
        for I in subsets_by_size[subset_size]:
            for i in I:
                proper_subset_cache[(I, i)] = proper_subsets_containing_i(I, i)
        if num_workers == 1 or len(current_subsets) == 1:
            worker_count = 1
        else:
            worker_count = min(num_workers, len(current_subsets))
        seeds = rng.integers(0, 2**63 - 1, size=len(current_subsets), dtype=np.int64)
        results = _run_subset_calibration_jobs(
            subsets=current_subsets,
            threshold=threshold,
            Sigma=Sigma,
            alpha=alpha,
            B=B,
            theta_alt=theta_alt,
            seeds=[int(seed) for seed in seeds],
            proper_subset_cache=proper_subset_cache,
            num_workers=worker_count,
        )
        for subset, subset_threshold in results:
            threshold[subset] = subset_threshold

    score_function, local_test = _build_callable_maps(
        threshold=threshold,
        Sigma=Sigma,
        theta_alt=theta_alt,
        subsets_by_size=subsets_by_size,
        proper_subset_cache=proper_subset_cache,
    )

    return BUProcedureFit(
        Sigma=Sigma,
        theta_alt=float(theta_alt),
        alpha=float(alpha),
        threshold=threshold,
        score_function=score_function,
        local_test=local_test,
        subsets_by_size=subsets_by_size,
        proper_subset_cache=proper_subset_cache,
        num_workers=num_workers,
    )


def apply_bu_procedure(p: Sequence[float], fit_object: BUProcedureFit) -> np.ndarray:
    p = _clip_pvalues(p)
    K = p.shape[0]
    if K != fit_object.Sigma.shape[0]:
        raise ValueError("Length of p must match the dimension of Sigma.")

    decisions = np.zeros(K, dtype=int)
    all_subsets: Iterable[Subset] = (
        subset
        for subset_size in range(1, K + 1)
        for subset in fit_object.subsets_by_size[subset_size]
    )
    subsets_list = list(all_subsets)

    for i in range(K):
        reject_i = 1
        for I in subsets_list:
            if i not in I:
                continue
            p_I = np.asarray([p[j] for j in I], dtype=float)
            if fit_object.local_test[I](p_I) == 0:
                reject_i = 0
                break
        decisions[i] = reject_i

    return decisions


def apply_hommel_procedure(p: Sequence[float], alpha: float = 0.05) -> np.ndarray:
    p_vec = _clip_pvalues(p)
    reject, _, _, _ = multipletests(p_vec, alpha=alpha, method="hommel")
    return reject.astype(int)


def _summarize_decisions(
    h_samples: np.ndarray,
    z_samples: np.ndarray,
    p_samples: np.ndarray,
    decisions: np.ndarray,
) -> Dict[str, np.ndarray | float]:
    def _se(values: np.ndarray) -> float:
        values = np.asarray(values, dtype=float)
        if values.size <= 1:
            return 0.0
        return float(np.std(values, ddof=1) / np.sqrt(values.size))

    n_sim = h_samples.shape[0]
    any_rejection = np.empty(n_sim, dtype=int)
    false_rejection = np.empty(n_sim, dtype=int)
    true_rejection = np.empty(n_sim, dtype=int)
    fdp = np.zeros(n_sim, dtype=float)
    tdp = np.zeros(n_sim, dtype=float)

    for sim_idx in range(n_sim):
        h_vec = h_samples[sim_idx]
        decision_vec = decisions[sim_idx]
        null_mask = h_vec == 0
        alt_mask = h_vec == 1
        num_rejections = int(decision_vec.sum())
        num_false = int(np.sum(decision_vec[null_mask])) if np.any(null_mask) else 0
        num_true = int(np.sum(decision_vec[alt_mask])) if np.any(alt_mask) else 0
        any_rejection[sim_idx] = int(num_rejections > 0)
        false_rejection[sim_idx] = int(num_false > 0)
        true_rejection[sim_idx] = int(num_true > 0)
        fdp[sim_idx] = num_false / num_rejections if num_rejections > 0 else 0.0
        if np.any(alt_mask):
            tdp[sim_idx] = num_true / int(np.sum(alt_mask))

    return {
        "h_samples": h_samples,
        "z_samples": z_samples,
        "p_samples": p_samples,
        "decisions": decisions,
        "per_hypothesis_rejection_rate": decisions.mean(axis=0),
        "fwer": float(false_rejection.mean()),
        "fwer_se": _se(false_rejection),
        "any_rejection_rate": float(any_rejection.mean()),
        "power_any": float(true_rejection.mean()),
        "power_any_se": _se(true_rejection),
        "average_power": float(np.mean(tdp)),
        "average_power_se": _se(tdp),
        "fdp_mean": float(np.mean(fdp)),
        "fdp_mean_se": _se(fdp),
        "false_rejection_indicator": false_rejection,
        "true_rejection_indicator": true_rejection,
    }


def simulate_bu_experiment(
    fit_object: BUProcedureFit,
    n_sim: int,
    random_seed: int | None = None,
    h: Sequence[int] | None = None,
    pi1: float = 0.5,
    n_jobs: int | None = None,
    num_workers: int | None = None,
) -> Dict[str, np.ndarray | float]:
    if n_sim <= 0:
        raise ValueError("n_sim must be a positive integer.")

    rng = np.random.default_rng(random_seed)
    K = fit_object.Sigma.shape[0]
    if num_workers is not None and n_jobs is not None and n_jobs != num_workers:
        raise ValueError("Pass only one of n_jobs or num_workers, or use matching values.")
    requested_workers = (
        num_workers
        if num_workers is not None
        else n_jobs if n_jobs is not None
        else fit_object.num_workers
    )
    worker_count = min(_resolve_num_workers(requested_workers), n_sim)
    if h is None:
        fixed_h = None
    else:
        fixed_h = _validate_configuration(h, K)

    if worker_count == 1:
        chunk_sizes = [n_sim]
    else:
        chunk_sizes = [size for size in _split_evenly(n_sim, worker_count) if size > 0]
    seeds = rng.integers(0, 2**63 - 1, size=len(chunk_sizes), dtype=np.int64)
    fit_payload = _fit_payload_from_object(fit_object)
    chunk_results = _run_simulation_jobs(
        fit_payload=fit_payload,
        chunk_sizes=chunk_sizes,
        seeds=[int(seed) for seed in seeds],
        h=fixed_h,
        pi1=pi1,
        num_workers=worker_count,
    )

    h_samples = np.vstack([chunk["h_samples"] for chunk in chunk_results])
    z_samples = np.vstack([chunk["z_samples"] for chunk in chunk_results])
    p_samples = np.vstack([chunk["p_samples"] for chunk in chunk_results])
    decisions = np.vstack([chunk["decisions"] for chunk in chunk_results])
    any_rejection = np.concatenate([chunk["any_rejection"] for chunk in chunk_results])
    false_rejection = np.concatenate([chunk["false_rejection"] for chunk in chunk_results])
    true_rejection = np.concatenate([chunk["true_rejection"] for chunk in chunk_results])
    fdp = np.concatenate([chunk["fdp"] for chunk in chunk_results])
    tdp = np.concatenate([chunk["tdp"] for chunk in chunk_results])

    return {
        "h_samples": h_samples,
        "z_samples": z_samples,
        "p_samples": p_samples,
        "decisions": decisions,
        "per_hypothesis_rejection_rate": decisions.mean(axis=0),
        "fwer": float(false_rejection.mean()),
        "fwer_se": float(np.std(false_rejection, ddof=1) / np.sqrt(false_rejection.size))
        if false_rejection.size > 1
        else 0.0,
        "any_rejection_rate": float(any_rejection.mean()),
        "power_any": float(true_rejection.mean()),
        "power_any_se": float(np.std(true_rejection, ddof=1) / np.sqrt(true_rejection.size))
        if true_rejection.size > 1
        else 0.0,
        "average_power": float(np.mean(tdp)),
        "average_power_se": float(np.std(tdp, ddof=1) / np.sqrt(tdp.size))
        if tdp.size > 1
        else 0.0,
        "fdp_mean": float(np.mean(fdp)),
        "fdp_mean_se": float(np.std(fdp, ddof=1) / np.sqrt(fdp.size))
        if fdp.size > 1
        else 0.0,
        "false_rejection_indicator": false_rejection,
        "true_rejection_indicator": true_rejection,
    }


def simulate_competing_procedures(
    fit_object: BUProcedureFit,
    n_sim: int,
    random_seed: int | None = None,
    h: Sequence[int] | None = None,
    pi1: float = 0.5,
    n_jobs: int | None = None,
    num_workers: int | None = None,
) -> Dict[str, Dict[str, np.ndarray | float]]:
    if n_sim <= 0:
        raise ValueError("n_sim must be a positive integer.")

    rng = np.random.default_rng(random_seed)
    K = fit_object.Sigma.shape[0]
    if num_workers is not None and n_jobs is not None and n_jobs != num_workers:
        raise ValueError("Pass only one of n_jobs or num_workers, or use matching values.")
    requested_workers = (
        num_workers
        if num_workers is not None
        else n_jobs if n_jobs is not None
        else fit_object.num_workers
    )
    worker_count = min(_resolve_num_workers(requested_workers), n_sim)

    if h is None:
        fixed_h = None
    else:
        fixed_h = _validate_configuration(h, K)

    if worker_count == 1:
        chunk_sizes = [n_sim]
    else:
        chunk_sizes = [size for size in _split_evenly(n_sim, worker_count) if size > 0]

    seeds = rng.integers(0, 2**63 - 1, size=len(chunk_sizes), dtype=np.int64)
    fit_payload = _fit_payload_from_object(fit_object)
    chunk_results = _run_competition_jobs(
        fit_payload=fit_payload,
        chunk_sizes=chunk_sizes,
        seeds=[int(seed) for seed in seeds],
        h=fixed_h,
        pi1=pi1,
        num_workers=worker_count,
    )

    h_samples = np.vstack([chunk["h_samples"] for chunk in chunk_results])
    z_samples = np.vstack([chunk["z_samples"] for chunk in chunk_results])
    p_samples = np.vstack([chunk["p_samples"] for chunk in chunk_results])
    bu_decisions = np.vstack([chunk["bu_decisions"] for chunk in chunk_results])
    hommel_decisions = np.vstack([chunk["hommel_decisions"] for chunk in chunk_results])

    return {
        "bu": _summarize_decisions(h_samples, z_samples, p_samples, bu_decisions),
        "hommel": _summarize_decisions(h_samples, z_samples, p_samples, hommel_decisions),
    }


def estimate_fwer(
    fit_object: BUProcedureFit,
    n_sim: int,
    random_seed: int | None = None,
    n_jobs: int | None = None,
    num_workers: int | None = None,
) -> Dict[str, np.ndarray | float]:
    K = fit_object.Sigma.shape[0]
    return simulate_bu_experiment(
        fit_object=fit_object,
        n_sim=n_sim,
        random_seed=random_seed,
        h=np.zeros(K, dtype=int),
        n_jobs=n_jobs,
        num_workers=num_workers,
    )


def estimate_power(
    fit_object: BUProcedureFit,
    n_sim: int,
    random_seed: int | None = None,
    h: Sequence[int] | None = None,
    pi1: float = 0.5,
    n_jobs: int | None = None,
    num_workers: int | None = None,
) -> Dict[str, np.ndarray | float]:
    result = simulate_bu_experiment(
        fit_object=fit_object,
        n_sim=n_sim,
        random_seed=random_seed,
        h=h,
        pi1=pi1,
        n_jobs=n_jobs,
        num_workers=num_workers,
    )
    if h is not None and np.sum(np.asarray(h, dtype=int)) == 0:
        raise ValueError("Power is undefined for an all-null fixed configuration.")
    return result


def estimate_fwer_competition(
    fit_object: BUProcedureFit,
    n_sim: int,
    random_seed: int | None = None,
    n_jobs: int | None = None,
    num_workers: int | None = None,
) -> Dict[str, Dict[str, np.ndarray | float]]:
    K = fit_object.Sigma.shape[0]
    return simulate_competing_procedures(
        fit_object=fit_object,
        n_sim=n_sim,
        random_seed=random_seed,
        h=np.zeros(K, dtype=int),
        n_jobs=n_jobs,
        num_workers=num_workers,
    )


def estimate_power_competition(
    fit_object: BUProcedureFit,
    n_sim: int,
    random_seed: int | None = None,
    h: Sequence[int] | None = None,
    pi1: float = 0.5,
    n_jobs: int | None = None,
    num_workers: int | None = None,
) -> Dict[str, Dict[str, np.ndarray | float]]:
    result = simulate_competing_procedures(
        fit_object=fit_object,
        n_sim=n_sim,
        random_seed=random_seed,
        h=h,
        pi1=pi1,
        n_jobs=n_jobs,
        num_workers=num_workers,
    )
    if h is not None and np.sum(np.asarray(h, dtype=int)) == 0:
        raise ValueError("Power is undefined for an all-null fixed configuration.")
    return result


def simulate_all_configurations(
    fit_object: BUProcedureFit,
    n_sim: int,
    random_seed: int | None = None,
    n_jobs: int | None = None,
    num_workers: int | None = None,
) -> Dict[Tuple[int, ...], Dict[str, np.ndarray | float]]:
    K = fit_object.Sigma.shape[0]
    configurations = all_binary_configurations(K)
    base_rng = np.random.default_rng(random_seed)
    seeds = base_rng.integers(0, 2**63 - 1, size=len(configurations), dtype=np.int64)

    results: Dict[Tuple[int, ...], Dict[str, np.ndarray | float]] = {}
    for h, seed in zip(configurations, seeds):
        results[h] = simulate_bu_experiment(
            fit_object=fit_object,
            n_sim=n_sim,
            random_seed=int(seed),
            h=h,
            n_jobs=n_jobs,
            num_workers=num_workers,
        )
    return results


def simulate_all_configurations_competition(
    fit_object: BUProcedureFit,
    n_sim: int,
    random_seed: int | None = None,
    n_jobs: int | None = None,
    num_workers: int | None = None,
) -> Dict[Tuple[int, ...], Dict[str, Dict[str, np.ndarray | float]]]:
    K = fit_object.Sigma.shape[0]
    configurations = all_binary_configurations(K)
    base_rng = np.random.default_rng(random_seed)
    seeds = base_rng.integers(0, 2**63 - 1, size=len(configurations), dtype=np.int64)

    results: Dict[Tuple[int, ...], Dict[str, Dict[str, np.ndarray | float]]] = {}
    for h, seed in zip(configurations, seeds):
        results[h] = simulate_competing_procedures(
            fit_object=fit_object,
            n_sim=n_sim,
            random_seed=int(seed),
            h=h,
            n_jobs=n_jobs,
            num_workers=num_workers,
        )
    return results
