import importlib
import subprocess
import sys
from statistics import NormalDist
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np


def _load_joblib_with_auto_install():
    try:
        joblib = importlib.import_module("joblib")
        return joblib.Parallel, joblib.delayed
    except ModuleNotFoundError:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "joblib"])
            joblib = importlib.import_module("joblib")
            print("Installed joblib successfully.")
            return joblib.Parallel, joblib.delayed
        except Exception:
            return None, None


Parallel, delayed = _load_joblib_with_auto_install()
STD_NORMAL = NormalDist(mu=0.0, sigma=1.0)


def _normal_cdf_vec(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.vectorize(STD_NORMAL.cdf, otypes=[float])(x)


def _sample_null_pvalues_independent(rng: np.random.Generator, n: int, k: int) -> np.ndarray:
    p = rng.uniform(0.0, 1.0, size=(n, k))
    p.sort(axis=1)
    return p


def _sample_null_pvalues_equicorrelated(
    rng: np.random.Generator,
    n: int,
    k: int,
    phi: float,
) -> np.ndarray:
    phi_min = -1.0 / (k - 1) if k > 1 else -1.0
    if not (phi_min <= phi < 1.0):
        raise ValueError(f"phi must be in [{phi_min}, 1).")
    cov = np.full((k, k), phi, dtype=float)
    np.fill_diagonal(cov, 1.0)
    z = rng.multivariate_normal(mean=np.zeros(k), cov=cov, size=n)
    p = _normal_cdf_vec(z)
    p.sort(axis=1)
    return p


def cms_bottomup_scores(
    p_sorted: Sequence[float],
    thresholds_t2_to_tk_minus_1: Sequence[float],
    f: Callable[[float], float],
    h: Callable[[float], float],
    alpha: float,
) -> Dict[str, List[np.ndarray]]:
    """
    Algorithm 1: Calculate k-dimensional CMS bottom-up scores.

    Inputs
    ------
    p_sorted:
        Sorted p-values p1 <= p2 <= ... <= pk.
    thresholds_t2_to_tk_minus_1:
        [t2, t3, ..., t_{k-1}] (length k-2 for k>=3).
    f, h:
        Functions used in Eq. (18): a_1^{|I|}(u_I) = f(u_k) * prod_i h(u_i).
    alpha:
        Significance level for phi_1(pl) = I{pl <= alpha}.

    Returns
    -------
    dict with:
      - final_scores:
            [s_k(p1..pk), s_{k-1}(p2..pk), ..., s_1(pk)]
      - score_rows:
            row l-1 stores all s_l(p_I) values for m=1..k-l+1
      - coeff_rows:
            row l-1 stores all a_1^{l}(p_I) values for m=1..k-l+1
      - phi_rows:
            row l-1 stores gating booleans used in recursion
    """
    p = np.asarray(p_sorted, dtype=float)
    if p.ndim != 1:
        raise ValueError("p_sorted must be a 1D array.")
    if np.any(np.diff(p) < 0):
        raise ValueError("p_sorted must be sorted in non-decreasing order.")

    k = p.size
    if k == 0:
        raise ValueError("p_sorted cannot be empty.")
    if k >= 3 and len(thresholds_t2_to_tk_minus_1) != (k - 2):
        raise ValueError("thresholds_t2_to_tk_minus_1 must have length k-2.")

    # Precompute f(p_i) and h(p_i) once for reuse throughout the recursion.
    f_vals = np.array([f(x) for x in p], dtype=float)
    h_vals = np.array([h(x) for x in p], dtype=float)

    # l = 1 (Algorithm line 1)
    a_prev = f_vals.copy()  # s1(pl)=a1^{ {l} }(pl)=f(pl)
    s_prev = a_prev.copy()
    phi_prev = (p <= alpha).astype(bool)  # phi1(pl)=I{pl<=alpha}

    coeff_rows: List[np.ndarray] = [a_prev.copy()]
    score_rows: List[np.ndarray] = [s_prev.copy()]
    phi_rows: List[np.ndarray] = [phi_prev.copy()]

    # l = 2..k
    for l in range(2, k + 1):
        q = k - l + 2  # second index in I = {m, q, ..., k} (1-based)
        n_cur = k - l + 1

        a_cur = np.zeros(n_cur, dtype=float)
        s_cur = np.zeros(n_cur, dtype=float)
        phi_cur = np.zeros(n_cur, dtype=bool)

        for m in range(1, n_cur + 1):
            i = m - 1
            q_idx = q - 1

            # Eq. (18), Algorithm line 6:
            # a_1^{|I|}(p_I) = a_1^{|J|}(p_J) * h(p_{k-l+2})
            a_cur[i] = a_prev[i] * h_vals[q_idx]

            # Eq. (19):
            # s_l(p_I) = I(s_{l-1}(p_{I\{q}})>t_{l-1}) * a_1^{|I|}(p_I)
            #          + h(p_m) * I(s_{l-1}(p_{I\{m}})>t_{l-1}) * s_{l-1}(p_{I\{m}})
            if l == 2:
                left_gate = phi_prev[i]
                right_gate = phi_prev[q_idx]
            else:
                t = thresholds_t2_to_tk_minus_1[l - 3]  # l=3 -> t2
                left_gate = s_prev[i] > t
                right_gate = s_prev[q_idx] > t

            tail_score = s_prev[q_idx]  # s_{l-1}(p_{I\{m}})
            s_cur[i] = float(left_gate) * a_cur[i] + h_vals[i] * float(right_gate) * tail_score
            phi_cur[i] = bool(left_gate and right_gate)

        coeff_rows.append(a_cur.copy())
        score_rows.append(s_cur.copy())
        phi_rows.append(phi_cur.copy())
        a_prev, s_prev, phi_prev = a_cur, s_cur, phi_cur

    # Algorithm line 10 output order:
    # s_k(p1..pk), s_{k-1}(p2..pk), ..., s_1(pk)
    final_scores = []
    for dim in range(k, 0, -1):
        row = score_rows[dim - 1]
        idx = k - dim  # m = k-dim+1 (0-based)
        final_scores.append(float(row[idx]))

    return {
        "final_scores": np.array(final_scores, dtype=float),
        "score_rows": score_rows,
        "coeff_rows": coeff_rows,
        "phi_rows": phi_rows,
    }


def _empirical_eq17_threshold(scores: np.ndarray, alpha: float, method: str = "higher") -> float:
    """
    Empirical approximation to Eq. (17):
        t_k = inf{ t : P(s_k > t) <= alpha }.

    For Monte Carlo samples, a conservative choice is an upper empirical quantile,
    implemented via np.quantile(..., method="higher") by default.
    """
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("scores must be a non-empty 1D array.")
    return float(np.quantile(scores, 1.0 - alpha, method=method))


def _top_dim_score_one_sample(
    p_sorted: np.ndarray,
    prev_thresholds: Sequence[float],
    f: Callable[[float], float],
    h: Callable[[float], float],
    alpha: float,
) -> float:
    out = cms_bottomup_scores(p_sorted, prev_thresholds, f, h, alpha)
    return float(out["final_scores"][0])


def estimate_cms_thresholds(
    K: int,
    B: int,
    f: Callable[[float], float],
    h: Callable[[float], float],
    alpha: float,
    seed: Optional[int] = None,
    quantile_method: str = "higher",
    chunk_size: Optional[int] = None,
    n_jobs: int = 1,
    null_correlation_phi: float = 0.0,
    verbose: bool = True,
) -> np.ndarray:
    """
    Algorithm 2 (Monte Carlo version) to estimate CMS bottom-up thresholds t_k, k=2..K.

    Parameters
    ----------
    K : int
        Maximum dimension.
    B : int
        Number of Monte Carlo samples used for each k.
    f, h, alpha
        Same as in cms_bottomup_scores.
    seed : int, optional
        RNG seed for reproducibility.
    quantile_method : str
        Passed to np.quantile. "higher" is conservative for Eq. (17) tail control.
    chunk_size : int, optional
        If provided, computes scores in chunks to reduce memory usage.
    n_jobs : int
        Number of workers for Monte Carlo scoring. Use -1 for all CPUs.
    null_correlation_phi : float
        If 0, sample iid Uniform(0,1) null p-values. Otherwise, sample
        equicorrelated Gaussian null statistics with correlation phi and
        transform them to one-sided lower-tail p-values.
    verbose : bool
        Print progress.

    Returns
    -------
    np.ndarray
        Thresholds [t2, t3, ..., tK].
    """
    if K < 2:
        raise ValueError("K must be >= 2.")
    if B <= 0:
        raise ValueError("B must be positive.")
    use_parallel = (n_jobs != 1) and (Parallel is not None) and (delayed is not None)
    if (n_jobs != 1) and not use_parallel and verbose:
        print("joblib not found; falling back to n_jobs=1.")

    rng = np.random.default_rng(seed)
    thresholds: List[float] = []

    for k in range(2, K + 1):
        if verbose:
            print(f"Estimating t_{k} with B={B} samples...")

        if chunk_size is None or chunk_size >= B:
            if null_correlation_phi == 0.0:
                p = _sample_null_pvalues_independent(rng, B, k)
            else:
                p = _sample_null_pvalues_equicorrelated(rng, B, k, null_correlation_phi)
            prev_thresholds = thresholds[: k - 2]  # [t2, ..., t_{k-1}]
            if use_parallel:
                scores = np.array(
                    Parallel(n_jobs=n_jobs, prefer="processes")(
                        delayed(_top_dim_score_one_sample)(p[b], prev_thresholds, f, h, alpha)
                        for b in range(B)
                    ),
                    dtype=float,
                )
            else:
                scores = np.empty(B, dtype=float)
                for b in range(B):
                    scores[b] = _top_dim_score_one_sample(p[b], prev_thresholds, f, h, alpha)
        else:
            scores_chunks: List[np.ndarray] = []
            prev_thresholds = thresholds[: k - 2]
            remaining = B
            while remaining > 0:
                n = min(chunk_size, remaining)
                if null_correlation_phi == 0.0:
                    p = _sample_null_pvalues_independent(rng, n, k)
                else:
                    p = _sample_null_pvalues_equicorrelated(rng, n, k, null_correlation_phi)
                if use_parallel:
                    chunk_scores = np.array(
                        Parallel(n_jobs=n_jobs, prefer="processes")(
                            delayed(_top_dim_score_one_sample)(p[b], prev_thresholds, f, h, alpha)
                            for b in range(n)
                        ),
                        dtype=float,
                    )
                else:
                    chunk_scores = np.empty(n, dtype=float)
                    for b in range(n):
                        chunk_scores[b] = _top_dim_score_one_sample(p[b], prev_thresholds, f, h, alpha)
                scores_chunks.append(chunk_scores)
                remaining -= n
            scores = np.concatenate(scores_chunks)

        t_k = _empirical_eq17_threshold(scores, alpha, method=quantile_method)
        thresholds.append(t_k)

        if verbose:
            tail_prob = float(np.mean(scores > t_k))
            print(f"t_{k} = {t_k:.8g}  (empirical P[s_{k} > t_{k}] = {tail_prob:.4f})")

    return np.array(thresholds, dtype=float)


def apply_cms_bottomup_testing(
    p_sorted: Sequence[float],
    thresholds_t2_to_tK: Sequence[float],
    f: Callable[[float], float],
    h: Callable[[float], float],
    alpha: float,
) -> Dict[str, np.ndarray]:
    """
    Algorithm 3: Apply CMS bottom-up testing procedure.

    Parameters
    ----------
    p_sorted
        Sorted p-values p1 <= ... <= pK.
    thresholds_t2_to_tK
        Thresholds [t2, ..., tK] (length K-1).
    f, h, alpha
        Same as Algorithm 1 / Eq. (18).

    Returns
    -------
    dict with:
      - decisions_sorted: D1(p), ..., DK(p) in sorted-p index order
      - phi_by_dim: [phi1, phi2, ..., phiK] where phi_d gates s_d
      - final_scores: [sK(p1..pK), s_{K-1}(p2..pK), ..., s1(pK)]
    """
    p = np.asarray(p_sorted, dtype=float)
    if np.any(np.diff(p) < 0):
        raise ValueError("p_sorted must be sorted in non-decreasing order.")
    K = p.size
    if K < 1:
        raise ValueError("p_sorted cannot be empty.")
    if K >= 2 and len(thresholds_t2_to_tK) != (K - 1):
        raise ValueError("thresholds_t2_to_tK must have length K-1 ([t2,...,tK]).")

    # Algorithm 1 needs thresholds [t2,...,t_{K-1}] only.
    alg1_thresholds = list(thresholds_t2_to_tK[:-1]) if K >= 3 else []
    out = cms_bottomup_scores(p, alg1_thresholds, f, h, alpha)
    final_scores = out["final_scores"]  # [sK, sK-1, ..., s1]

    # Build phi_d for d=1..K.
    phi = np.zeros(K, dtype=int)
    # Consistent with Algorithm 1 initialization.
    phi[0] = int(p[-1] <= alpha)  # phi1(pK)
    for d in range(2, K + 1):
        score_d = final_scores[K - d]     # map dim d -> score entry
        t_d = thresholds_t2_to_tK[d - 2]  # t2 at index 0
        phi[d - 1] = int(score_d > t_d)

    # Algorithm 3 recursion:
    # D1 = phiK ; D_r = D_{r-1} * phi_{K-r+1}, r=2..K.
    decisions = np.zeros(K, dtype=int)
    decisions[0] = phi[K - 1]
    for r in range(2, K + 1):
        decisions[r - 1] = decisions[r - 2] * phi[K - r]

    return {
        "decisions_sorted": decisions,
        "phi_by_dim": phi,
        "final_scores": final_scores,
    }


if __name__ == "__main__":
    # Example usage with simple f, h.
    p = np.array([0.001, 0.01, 0.03, 0.07])  # sorted
    t = [0.2, 0.4]  # [t2, t3] for k=4
    alpha = 0.05

    def f(u: float) -> float:
        return np.exp(-u)

    def h(u: float) -> float:
        return 1.0 + np.exp(-u)

    out = cms_bottomup_scores(p, t, f, h, alpha)
    print("final_scores:", out["final_scores"])

    # Algorithm 2 example (small B for demo)
    th = estimate_cms_thresholds(K=4, B=2000, f=f, h=h, alpha=alpha, seed=123, chunk_size=500)
    print("thresholds [t2..tK]:", th)

    # Algorithm 3 example
    dec = apply_cms_bottomup_testing(p, thresholds_t2_to_tK=th, f=f, h=h, alpha=alpha)
    print("phi_by_dim [phi1..phiK]:", dec["phi_by_dim"])
    print("decisions_sorted [D1..DK]:", dec["decisions_sorted"])
