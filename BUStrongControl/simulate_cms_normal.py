import argparse
import importlib
import json
import os
import subprocess
import sys
import time
from statistics import NormalDist
from typing import Dict, List, Optional

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


def _load_pyplot():
    try:
        plt = importlib.import_module("matplotlib.pyplot")
        return plt
    except ModuleNotFoundError:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
            plt = importlib.import_module("matplotlib.pyplot")
            print("Installed matplotlib successfully.")
            return plt
        except Exception:
            return None
    except Exception:
        return None


plt = _load_pyplot()

from cms_bottomup import apply_cms_bottomup_testing, estimate_cms_thresholds
from gou_gtxr import gou_gtxr_batch

STD_NORMAL = NormalDist(mu=0.0, sigma=1.0)


def procedure_runtime_clock() -> float:
    """CPU-time clock used for comparable per-procedure runtime measurements."""
    return time.process_time()


def normal_cdf(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.vectorize(STD_NORMAL.cdf, otypes=[float])(x)


def normal_ppf(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    return np.vectorize(STD_NORMAL.inv_cdf, otypes=[float])(u)


def g_theta_normal_p(u: np.ndarray, theta: float) -> np.ndarray:
    u = np.clip(np.asarray(u, dtype=float), 1e-15, 1 - 1e-15)
    return np.exp(theta * normal_ppf(u) - 0.5 * theta**2)


def beta_a1_density(u: np.ndarray, a: float) -> np.ndarray:
    u = np.clip(np.asarray(u, dtype=float), 1e-15, 1.0)
    return a * (u ** (a - 1.0))


def beta_a1_ppf(u: np.ndarray, a: float) -> np.ndarray:
    u = np.clip(np.asarray(u, dtype=float), 1e-15, 1 - 1e-15)
    return u ** (1.0 / a)


def apply_procedure_one_sample(
    pvals: np.ndarray,
    thresholds_t2_to_tk: np.ndarray,
    f,
    h,
    alpha: float,
) -> np.ndarray:
    order = np.argsort(pvals)
    p_sorted = pvals[order]
    out = apply_cms_bottomup_testing(
        p_sorted=p_sorted,
        thresholds_t2_to_tK=thresholds_t2_to_tk,
        f=f,
        h=h,
        alpha=alpha,
    )
    d_sorted = out["decisions_sorted"].astype(int)
    d = np.zeros_like(d_sorted)
    d[order] = d_sorted
    return d


def _hommel_convex_hull(p_sorted: np.ndarray) -> List[int]:
    """Return the one-based hull indices used by the fast Hommel algorithm."""
    m = p_sorted.size
    hull = [0, 1]

    for i in range(2, m + 1):
        if i == m or (m - 1) * (p_sorted[i - 1] - p_sorted[0]) < (i - 1) * (
            p_sorted[m - 1] - p_sorted[0]
        ):
            while True:
                r = len(hull) - 1
                if r > 1:
                    not_convex = (i - hull[r - 1]) * (
                        p_sorted[hull[r] - 1] - p_sorted[hull[r - 1] - 1]
                    ) >= (hull[r] - hull[r - 1]) * (
                        p_sorted[i - 1] - p_sorted[hull[r - 1] - 1]
                    )
                elif r == 1:
                    not_convex = i * p_sorted[hull[1] - 1] >= hull[1] * p_sorted[i - 1]
                else:
                    not_convex = False

                if not not_convex:
                    break
                hull.pop()

            hull.append(i)

    return hull


def hommel_adjusted_pvalues(pvals: np.ndarray) -> np.ndarray:
    """Compute Hommel-adjusted p-values using the fast hommel-package algorithm."""
    p = np.asarray(pvals, dtype=float)
    if p.ndim != 1:
        raise ValueError("pvals must be a 1D array.")
    if p.size == 0:
        raise ValueError("pvals must not be empty.")
    if np.any(~np.isfinite(p)) or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("pvals must contain finite values in [0, 1].")
    if p.size == 1:
        return p.copy()

    order = np.argsort(p, kind="stable")
    p_sorted = p[order]
    m = p_sorted.size
    simes_factor = np.arange(m + 1, dtype=float)

    hull = _hommel_convex_hull(p_sorted)
    jump_alpha = np.zeros(m + 1, dtype=float)
    k = len(hull) - 1
    i = 1
    while i <= m:
        if k > 1:
            d_k = p_sorted[hull[k - 1] - 1] * (hull[k] - m + i) - p_sorted[
                hull[k] - 1
            ] * (hull[k - 1] - m + i)
        else:
            d_k = 0.0

        if k > 1 and d_k < 0.0:
            k -= 1
        else:
            jump_alpha[i - 1] = (
                simes_factor[i]
                * p_sorted[hull[k] - 1]
                / (hull[k] - m + i)
            )
            i += 1

    adjusted_sorted = np.zeros(m, dtype=float)
    i = 1
    j = m + 1
    while i <= m:
        if simes_factor[j - 1] * p_sorted[i - 1] <= jump_alpha[j - 1]:
            if j > m:
                adjusted_sorted[i - 1] = jump_alpha[m]
            else:
                adjusted_sorted[i - 1] = min(
                    simes_factor[j] * p_sorted[i - 1], jump_alpha[j - 1]
                )
            i += 1
        else:
            j -= 1

    adjusted = np.empty(m, dtype=float)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


def apply_hommel_one_sample(pvals: np.ndarray, alpha: float) -> np.ndarray:
    return (hommel_adjusted_pvalues(pvals) <= alpha).astype(int)


def apply_holm_one_sample(pvals: np.ndarray, alpha: float) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    if p.ndim != 1:
        raise ValueError("pvals must be a 1D array.")

    order = np.argsort(p)
    p_sorted = p[order]
    ntests = p_sorted.size

    cutoff = alpha / np.arange(ntests, 0, -1, dtype=float)
    reject_sorted = p_sorted <= cutoff

    first_nonreject = np.flatnonzero(~reject_sorted)
    if first_nonreject.size > 0:
        reject_sorted[first_nonreject[0]:] = False

    reject = np.zeros(ntests, dtype=int)
    reject[order] = reject_sorted.astype(int)
    return reject


def _decision_summary(decisions: np.ndarray, true_null: np.ndarray, false_mask: np.ndarray, n_false: int) -> tuple:
    any_true = int(np.any(decisions[true_null] == 1))
    if n_false > 0:
        false_rej = decisions[false_mask] == 1
        any_false = int(np.any(false_rej))
        tpr = float(false_rej.mean())
    else:
        any_false = 0
        tpr = float("nan")
    return any_true, any_false, tpr


def _metrics_from_summary(any_true_rej: np.ndarray, any_false_rej: np.ndarray, tpr: np.ndarray, n_false: int) -> Dict[str, float]:
    return {
        "FWER": float(any_true_rej.mean()),
        "Power_any": float(any_false_rej.mean()) if n_false > 0 else float("nan"),
        "Power_avg": float(np.nanmean(tpr)) if n_false > 0 else float("nan"),
    }


def _simulate_one_from_z(
    z: np.ndarray,
    true_null: np.ndarray,
    false_mask: np.ndarray,
    n_false: int,
    thresholds_t2_to_tk: np.ndarray,
    f,
    h,
    alpha: float,
) -> tuple:
    p = normal_cdf(z)  # one-sided lower-tail p-values
    d_cms = apply_procedure_one_sample(p, thresholds_t2_to_tk, f, h, alpha)
    any_true_cms = int(np.any(d_cms[true_null] == 1))
    d_holm = apply_holm_one_sample(p, alpha)
    any_true_holm = int(np.any(d_holm[true_null] == 1))
    d_hommel = apply_hommel_one_sample(p, alpha)
    any_true_hommel = int(np.any(d_hommel[true_null] == 1))

    if n_false > 0:
        false_rej_cms = d_cms[false_mask] == 1
        any_false_cms = int(np.any(false_rej_cms))
        tpr_cms = float(false_rej_cms.mean())
        false_rej_holm = d_holm[false_mask] == 1
        any_false_holm = int(np.any(false_rej_holm))
        tpr_holm = float(false_rej_holm.mean())
        false_rej_hommel = d_hommel[false_mask] == 1
        any_false_hommel = int(np.any(false_rej_hommel))
        tpr_hommel = float(false_rej_hommel.mean())
    else:
        any_false_cms = 0
        tpr_cms = float("nan")
        any_false_holm = 0
        tpr_holm = float("nan")
        any_false_hommel = 0
        tpr_hommel = float("nan")
    return (
        any_true_cms,
        any_false_cms,
        tpr_cms,
        any_true_holm,
        any_false_holm,
        tpr_holm,
        any_true_hommel,
        any_false_hommel,
        tpr_hommel,
    )


def simulate_config(
    n_sim: int,
    mu: np.ndarray,
    true_null: np.ndarray,
    rho: float,
    thresholds_t2_to_tk: np.ndarray,
    f,
    h,
    alpha: float,
    rng: np.random.Generator,
    n_jobs: int = 1,
) -> Dict[str, float]:
    K = mu.size
    cms_any_true_rej = np.zeros(n_sim, dtype=bool)
    cms_any_false_rej = np.zeros(n_sim, dtype=bool)
    cms_tpr = np.zeros(n_sim, dtype=float)
    holm_any_true_rej = np.zeros(n_sim, dtype=bool)
    holm_any_false_rej = np.zeros(n_sim, dtype=bool)
    holm_tpr = np.zeros(n_sim, dtype=float)
    hommel_any_true_rej = np.zeros(n_sim, dtype=bool)
    hommel_any_false_rej = np.zeros(n_sim, dtype=bool)
    hommel_tpr = np.zeros(n_sim, dtype=float)
    gtxr_any_true_rej = np.zeros(n_sim, dtype=bool)
    gtxr_any_false_rej = np.zeros(n_sim, dtype=bool)
    gtxr_tpr = np.zeros(n_sim, dtype=float)

    false_mask = ~true_null
    n_false = int(false_mask.sum())
    rho_min = -1.0 / (K - 1) if K > 1 else -1.0
    if not (rho_min <= rho < 1.0):
        raise ValueError(f"rho must be in [{rho_min}, 1).")

    cov = np.full((K, K), rho, dtype=float)
    np.fill_diagonal(cov, 1.0)
    z_all = rng.multivariate_normal(mean=mu, cov=cov, size=n_sim)
    p_all = normal_cdf(z_all)

    gtxr_decisions = gou_gtxr_batch(p_all, alpha)
    for i in range(n_sim):
        d_gtxr = gtxr_decisions[i]
        gtxr_any_true_rej[i] = bool(np.any(d_gtxr[true_null] == 1))
        if n_false > 0:
            false_rej_gtxr = d_gtxr[false_mask] == 1
            gtxr_any_false_rej[i] = bool(np.any(false_rej_gtxr))
            gtxr_tpr[i] = float(false_rej_gtxr.mean())
        else:
            gtxr_any_false_rej[i] = False
            gtxr_tpr[i] = np.nan

    use_parallel = (n_jobs != 1) and (Parallel is not None) and (delayed is not None)
    if use_parallel:
        vals = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_simulate_one_from_z)(
                z_all[s],
                true_null,
                false_mask,
                n_false,
                thresholds_t2_to_tk,
                f,
                h,
                alpha,
            )
            for s in range(n_sim)
        )
        for i, val in enumerate(vals):
            (
                cms_a_true,
                cms_a_false,
                cms_tpr_i,
                holm_a_true,
                holm_a_false,
                holm_tpr_i,
                hommel_a_true,
                hommel_a_false,
                hommel_tpr_i,
            ) = val
            cms_any_true_rej[i] = bool(cms_a_true)
            cms_any_false_rej[i] = bool(cms_a_false)
            cms_tpr[i] = cms_tpr_i
            holm_any_true_rej[i] = bool(holm_a_true)
            holm_any_false_rej[i] = bool(holm_a_false)
            holm_tpr[i] = holm_tpr_i
            hommel_any_true_rej[i] = bool(hommel_a_true)
            hommel_any_false_rej[i] = bool(hommel_a_false)
            hommel_tpr[i] = hommel_tpr_i
    else:
        for s in range(n_sim):
            (
                cms_a_true,
                cms_a_false,
                cms_tpr_i,
                holm_a_true,
                holm_a_false,
                holm_tpr_i,
                hommel_a_true,
                hommel_a_false,
                hommel_tpr_i,
            ) = _simulate_one_from_z(
                z_all[s], true_null, false_mask, n_false, thresholds_t2_to_tk, f, h, alpha
            )
            cms_any_true_rej[s] = bool(cms_a_true)
            cms_any_false_rej[s] = bool(cms_a_false)
            cms_tpr[s] = cms_tpr_i
            holm_any_true_rej[s] = bool(holm_a_true)
            holm_any_false_rej[s] = bool(holm_a_false)
            holm_tpr[s] = holm_tpr_i
            hommel_any_true_rej[s] = bool(hommel_a_true)
            hommel_any_false_rej[s] = bool(hommel_a_false)
            hommel_tpr[s] = hommel_tpr_i

    out = {
        "CMS_FWER": float(cms_any_true_rej.mean()),
        "CMS_Power_any": float(cms_any_false_rej.mean()) if n_false > 0 else float("nan"),
        "CMS_Power_avg": float(np.nanmean(cms_tpr)) if n_false > 0 else float("nan"),
        "Holm_FWER": float(holm_any_true_rej.mean()),
        "Holm_Power_any": float(holm_any_false_rej.mean()) if n_false > 0 else float("nan"),
        "Holm_Power_avg": float(np.nanmean(holm_tpr)) if n_false > 0 else float("nan"),
        "GTXR_FWER": float(gtxr_any_true_rej.mean()),
        "GTXR_Power_any": float(gtxr_any_false_rej.mean()) if n_false > 0 else float("nan"),
        "GTXR_Power_avg": float(np.nanmean(gtxr_tpr)) if n_false > 0 else float("nan"),
    }
    out.update(
        {
            "Hommel_FWER": float(hommel_any_true_rej.mean()),
            "Hommel_Power_any": float(hommel_any_false_rej.mean()) if n_false > 0 else float("nan"),
            "Hommel_Power_avg": float(np.nanmean(hommel_tpr)) if n_false > 0 else float("nan"),
        }
    )
    return out


def simulate_config_beta_a1(
    n_sim: int,
    K: int,
    true_null: np.ndarray,
    rho: float,
    a: float,
    thresholds_t2_to_tk: np.ndarray,
    f,
    h,
    alpha: float,
    rng: np.random.Generator,
    n_jobs: int = 1,
) -> Dict[str, float]:
    false_mask = ~true_null
    n_false = int(false_mask.sum())
    rho_min = -1.0 / (K - 1) if K > 1 else -1.0
    if not (rho_min <= rho < 1.0):
        raise ValueError(f"rho must be in [{rho_min}, 1).")

    cov = np.full((K, K), rho, dtype=float)
    np.fill_diagonal(cov, 1.0)
    z_all = rng.multivariate_normal(mean=np.zeros(K), cov=cov, size=n_sim)
    u_all = normal_cdf(z_all)

    p_all = np.empty_like(u_all)
    p_all[:, true_null] = u_all[:, true_null]
    if n_false > 0:
        p_all[:, false_mask] = beta_a1_ppf(u_all[:, false_mask], a)

    cms_any_true_rej = np.zeros(n_sim, dtype=bool)
    cms_any_false_rej = np.zeros(n_sim, dtype=bool)
    cms_tpr = np.zeros(n_sim, dtype=float)
    holm_any_true_rej = np.zeros(n_sim, dtype=bool)
    holm_any_false_rej = np.zeros(n_sim, dtype=bool)
    holm_tpr = np.zeros(n_sim, dtype=float)
    hommel_any_true_rej = np.zeros(n_sim, dtype=bool)
    hommel_any_false_rej = np.zeros(n_sim, dtype=bool)
    hommel_tpr = np.zeros(n_sim, dtype=float)
    gtxr_any_true_rej = np.zeros(n_sim, dtype=bool)
    gtxr_any_false_rej = np.zeros(n_sim, dtype=bool)
    gtxr_tpr = np.zeros(n_sim, dtype=float)

    gtxr_decisions = gou_gtxr_batch(p_all, alpha)
    for i in range(n_sim):
        p = p_all[i]

        d_cms = apply_procedure_one_sample(p, thresholds_t2_to_tk, f, h, alpha)
        d_gtxr = gtxr_decisions[i]

        cms_any_true_rej[i] = bool(np.any(d_cms[true_null] == 1))
        gtxr_any_true_rej[i] = bool(np.any(d_gtxr[true_null] == 1))

        d_holm = apply_holm_one_sample(p, alpha)
        d_hommel = apply_hommel_one_sample(p, alpha)
        holm_any_true_rej[i] = bool(np.any(d_holm[true_null] == 1))
        hommel_any_true_rej[i] = bool(np.any(d_hommel[true_null] == 1))

        if n_false > 0:
            cms_false = d_cms[false_mask] == 1
            gtxr_false = d_gtxr[false_mask] == 1
            cms_any_false_rej[i] = bool(np.any(cms_false))
            gtxr_any_false_rej[i] = bool(np.any(gtxr_false))
            cms_tpr[i] = float(cms_false.mean())
            gtxr_tpr[i] = float(gtxr_false.mean())

            holm_false = d_holm[false_mask] == 1
            hommel_false = d_hommel[false_mask] == 1
            holm_any_false_rej[i] = bool(np.any(holm_false))
            hommel_any_false_rej[i] = bool(np.any(hommel_false))
            holm_tpr[i] = float(holm_false.mean())
            hommel_tpr[i] = float(hommel_false.mean())
        else:
            cms_tpr[i] = np.nan
            gtxr_tpr[i] = np.nan
            holm_tpr[i] = np.nan
            hommel_tpr[i] = np.nan

    out = {
        "CMS_FWER": float(cms_any_true_rej.mean()),
        "CMS_Power_any": float(cms_any_false_rej.mean()) if n_false > 0 else float("nan"),
        "CMS_Power_avg": float(np.nanmean(cms_tpr)) if n_false > 0 else float("nan"),
        "Holm_FWER": float(holm_any_true_rej.mean()),
        "Holm_Power_any": float(holm_any_false_rej.mean()) if n_false > 0 else float("nan"),
        "Holm_Power_avg": float(np.nanmean(holm_tpr)) if n_false > 0 else float("nan"),
        "GTXR_FWER": float(gtxr_any_true_rej.mean()),
        "GTXR_Power_any": float(gtxr_any_false_rej.mean()) if n_false > 0 else float("nan"),
        "GTXR_Power_avg": float(np.nanmean(gtxr_tpr)) if n_false > 0 else float("nan"),
        "Hommel_FWER": float(hommel_any_true_rej.mean()),
        "Hommel_Power_any": float(hommel_any_false_rej.mean()) if n_false > 0 else float("nan"),
        "Hommel_Power_avg": float(np.nanmean(hommel_tpr)) if n_false > 0 else float("nan"),
    }
    return out


def evaluate_methods_one_p(
    p: np.ndarray,
    true_null: np.ndarray,
    false_mask: np.ndarray,
    n_false: int,
    alpha: float,
    bu_procedures: List[Dict],
) -> Dict[str, Dict[str, float]]:
    out = {}
    for procedure in bu_procedures:
        start = procedure_runtime_clock()
        d_bu = apply_procedure_one_sample(p, procedure["thresholds"], procedure["f"], procedure["h"], alpha)
        elapsed = procedure_runtime_clock() - start
        out[procedure["name"]] = {
            "summary": _decision_summary(d_bu, true_null, false_mask, n_false),
            "time_seconds": elapsed,
        }

    start = procedure_runtime_clock()
    d_gtxr = gou_gtxr_batch(p[None, :], alpha)[0]
    elapsed = procedure_runtime_clock() - start
    out["GTXR"] = {
        "summary": _decision_summary(d_gtxr, true_null, false_mask, n_false),
        "time_seconds": elapsed,
    }

    start = procedure_runtime_clock()
    d_holm = apply_holm_one_sample(p, alpha)
    elapsed_holm = procedure_runtime_clock() - start
    start = procedure_runtime_clock()
    d_hommel = apply_hommel_one_sample(p, alpha)
    elapsed_hommel = procedure_runtime_clock() - start
    out["Holm"] = {
        "summary": _decision_summary(d_holm, true_null, false_mask, n_false),
        "time_seconds": elapsed_holm,
    }
    out["Hommel"] = {
        "summary": _decision_summary(d_hommel, true_null, false_mask, n_false),
        "time_seconds": elapsed_hommel,
    }
    return out


def _summarize_pvalue_matrix_multi_bu(
    p_all: np.ndarray,
    true_null: np.ndarray,
    alpha: float,
    bu_procedures: List[Dict],
    n_jobs: int = 1,
) -> Dict[str, float]:
    n_sim = p_all.shape[0]
    false_mask = ~true_null
    n_false = int(false_mask.sum())
    method_names = [procedure["name"] for procedure in bu_procedures] + [
        "GTXR",
        "Holm",
        "Hommel",
    ]

    summary = {
        name: {
            "any_true": np.zeros(n_sim, dtype=bool),
            "any_false": np.zeros(n_sim, dtype=bool),
            "tpr": np.zeros(n_sim, dtype=float),
            "time_seconds_total": 0.0,
        }
        for name in method_names
    }

    use_parallel = (n_jobs != 1) and (Parallel is not None) and (delayed is not None)
    if use_parallel:
        vals = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(evaluate_methods_one_p)(
                p_all[s],
                true_null,
                false_mask,
                n_false,
                alpha,
                bu_procedures,
            )
            for s in range(n_sim)
        )
        for i, row in enumerate(vals):
            for name, method_out in row.items():
                a_true, a_false, tpr_val = method_out["summary"]
                summary[name]["any_true"][i] = bool(a_true)
                summary[name]["any_false"][i] = bool(a_false)
                summary[name]["tpr"][i] = tpr_val
                summary[name]["time_seconds_total"] += float(method_out["time_seconds"])
    else:
        for i in range(n_sim):
            row = evaluate_methods_one_p(p_all[i], true_null, false_mask, n_false, alpha, bu_procedures)
            for name, method_out in row.items():
                a_true, a_false, tpr_val = method_out["summary"]
                summary[name]["any_true"][i] = bool(a_true)
                summary[name]["any_false"][i] = bool(a_false)
                summary[name]["tpr"][i] = tpr_val
                summary[name]["time_seconds_total"] += float(method_out["time_seconds"])

    out = {}
    for name in method_names:
        metrics = _metrics_from_summary(
            summary[name]["any_true"],
            summary[name]["any_false"],
            summary[name]["tpr"],
            n_false,
        )
        for metric_name, metric_val in metrics.items():
            out[f"{name}_{metric_name}"] = metric_val
        out[f"{name}_Time_total_seconds"] = float(summary[name]["time_seconds_total"])
        out[f"{name}_Time_avg_seconds"] = float(summary[name]["time_seconds_total"] / n_sim)
    return out


def simulate_config_multi_bu(
    n_sim: int,
    mu: np.ndarray,
    true_null: np.ndarray,
    rho: float,
    bu_procedures: List[Dict],
    alpha: float,
    rng: np.random.Generator,
    n_jobs: int = 1,
) -> Dict[str, float]:
    K = mu.size
    rho_min = -1.0 / (K - 1) if K > 1 else -1.0
    if not (rho_min <= rho < 1.0):
        raise ValueError(f"rho must be in [{rho_min}, 1).")

    cov = np.full((K, K), rho, dtype=float)
    np.fill_diagonal(cov, 1.0)
    z_all = rng.multivariate_normal(mean=mu, cov=cov, size=n_sim)
    p_all = normal_cdf(z_all)
    return _summarize_pvalue_matrix_multi_bu(p_all, true_null, alpha, bu_procedures, n_jobs=n_jobs)


def simulate_config_beta_a1_multi_bu(
    n_sim: int,
    K: int,
    true_null: np.ndarray,
    rho: float,
    a: float,
    bu_procedures: List[Dict],
    alpha: float,
    rng: np.random.Generator,
    n_jobs: int = 1,
) -> Dict[str, float]:
    rho_min = -1.0 / (K - 1) if K > 1 else -1.0
    if not (rho_min <= rho < 1.0):
        raise ValueError(f"rho must be in [{rho_min}, 1).")

    false_mask = ~true_null
    cov = np.full((K, K), rho, dtype=float)
    np.fill_diagonal(cov, 1.0)
    z_all = rng.multivariate_normal(mean=np.zeros(K), cov=cov, size=n_sim)
    u_all = normal_cdf(z_all)
    p_all = np.empty_like(u_all)
    p_all[:, true_null] = u_all[:, true_null]
    if np.any(false_mask):
        p_all[:, false_mask] = beta_a1_ppf(u_all[:, false_mask], a)
    return _summarize_pvalue_matrix_multi_bu(p_all, true_null, alpha, bu_procedures, n_jobs=n_jobs)


def build_default_configs(K: int) -> List[np.ndarray]:
    # config m: first m hypotheses are false (non-null), remaining true null.
    # This spans all counts of false hypotheses from 0..K.
    configs = []
    for m in range(0, K + 1):
        c = np.zeros(K, dtype=bool)
        c[m:] = True  # True means null hypothesis is true
        configs.append(c)
    return configs


def apply_all_procedures_to_pvalues(
    pvals: np.ndarray,
    threshold_file: str,
) -> Dict[str, object]:
    p = np.asarray(pvals, dtype=float)
    if p.ndim != 1:
        raise ValueError("pvals must be a 1D array.")
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("pvals must lie in [0, 1].")

    with open(threshold_file, "r", encoding="utf-8") as f_in:
        threshold_payload = json.load(f_in)

    params = threshold_payload["params"]
    bu_specs = threshold_payload["bu_procedures"]

    K = int(params["K"])
    alpha = float(params["alpha"])
    alternative = params.get("alternative", "normal")
    beta_a = float(params.get("beta_a", 0.3))

    if p.size != K:
        raise ValueError(f"pvals must have length {K}.")

    std = NormalDist()
    order = np.argsort(p)
    p_sorted = p[order]

    bu_results = {}
    for spec in bu_specs:
        objective = spec["objective"]
        target_power = float(spec["target_power"])
        thresholds = np.array(spec["thresholds_t2_to_tK"], dtype=float)

        if alternative == "normal":
            theta = std.inv_cdf(alpha / K) - std.inv_cdf(target_power)

            def f(u, theta=theta):
                return g_theta_normal_p(u, theta)

            if objective == "bayes":
                def h(u, theta=theta):
                    return 1.0 + g_theta_normal_p(u, theta)
            else:
                def h(u):
                    return 1.0
        elif alternative == "beta_a1":
            def f(u):
                return beta_a1_density(u, beta_a)

            if objective == "bayes":
                def h(u):
                    return 1.0 + beta_a1_density(u, beta_a)
            else:
                def h(u):
                    return 1.0
        else:
            raise ValueError("alternative must be 'normal' or 'beta_a1'")

        out = apply_cms_bottomup_testing(
            p_sorted=p_sorted,
            thresholds_t2_to_tK=thresholds,
            f=f,
            h=h,
            alpha=alpha,
        )

        d_sorted = out["decisions_sorted"].astype(int)
        d = np.zeros_like(d_sorted)
        d[order] = d_sorted

        bu_results[spec["name"]] = {
            "objective": objective,
            "target_power": target_power,
            "decisions": d,
            "scores": out["final_scores"],
            "phi": out["phi_by_dim"],
        }

    result = {
        "pvals": p,
        "BU": bu_results,
        "GTXR": gou_gtxr_batch(p[None, :], alpha)[0],
    }

    result["Holm"] = apply_holm_one_sample(p, alpha)
    result["Hommel"] = apply_hommel_one_sample(p, alpha)

    return result


def _format_method_label(method: str) -> str:
    if method.startswith("BU_bayes_tp_"):
        tp = method.removeprefix("BU_bayes_tp_").replace("p", ".")
        return rf"BU $\Pi_{{\mathrm{{mix}}}}({tp})$"
    if method.startswith("BU_pi1_tp_"):
        tp = method.removeprefix("BU_pi1_tp_").replace("p", ".")
        return rf"BU $\Pi_{{1}}({tp})$"
    if method == "GTXR":
        return "Gou"
    return method


def plot_simulation_results(results: List[Dict[str, float]], output_prefix: str) -> List[str]:
    if plt is None:
        print("matplotlib not found; skipping plot generation.")
        return []

    fwer_keys = [key for key in results[0].keys() if key.endswith("_FWER")]
    method_names = [key[:-5] for key in fwer_keys]
    time_avg_keys = [key for key in results[0].keys() if key.endswith("_Time_avg_seconds")]
    color_cycle = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    ]
    method_specs = [(name, _format_method_label(name), color_cycle[i % len(color_cycle)]) for i, name in enumerate(method_names)]

    fwer_rows = sorted((row for row in results if row["n_false"] <= row["K"] - 1), key=lambda r: r["n_false"])
    power_rows = sorted((row for row in results if row["n_false"] >= 1), key=lambda r: r["n_false"])

    paths = []

    fig, ax = plt.subplots(figsize=(10, 5))
    x_fwer = [row["n_false"] for row in fwer_rows]
    for method, label, color in method_specs:
        y = [row.get(f"{method}_FWER", float("nan")) for row in fwer_rows]
        ax.plot(x_fwer, y, marker="o", linewidth=2, color=color, label=label)
    ax.set_xlabel("Number of false hypotheses")
    ax.set_ylabel("FWER")
    ax.set_title("FWER by Number of False Hypotheses")
    ax.set_xticks(x_fwer)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    fig.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))
    fwer_path = f"{output_prefix}_fwer.png"
    fig.savefig(fwer_path, dpi=150)
    plt.close(fig)
    paths.append(fwer_path)

    fig, ax = plt.subplots(figsize=(10, 5))
    x_power = [row["n_false"] for row in power_rows]
    for method, label, color in method_specs:
        y = [row.get(f"{method}_Power_avg", float("nan")) for row in power_rows]
        ax.plot(x_power, y, marker="o", linewidth=2, color=color, label=label)
    ax.set_xlabel("Number of false hypotheses")
    ax.set_ylabel("Average power")
    ax.set_title("Power by Number of False Hypotheses")
    ax.set_xticks(x_power)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    fig.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))
    power_path = f"{output_prefix}_power.png"
    fig.savefig(power_path, dpi=150)
    plt.close(fig)
    paths.append(power_path)

    if time_avg_keys:
        fig, ax = plt.subplots(figsize=(10, 5))
        x_time = [row["n_false"] for row in results]
        for method, label, color in method_specs:
            y = [row.get(f"{method}_Time_avg_seconds", float("nan")) for row in results]
            ax.plot(x_time, y, marker="o", linewidth=2, color=color, label=label)
        ax.set_xlabel("Number of false hypotheses")
        ax.set_ylabel("Average CPU time per dataset (seconds)")
        ax.set_title("CPU Time by Number of False Hypotheses")
        ax.set_xticks(x_time)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
        fig.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))
        time_path = f"{output_prefix}_time.png"
        fig.savefig(time_path, dpi=150)
        plt.close(fig)
        paths.append(time_path)

    # Also write a single multi-panel figure with one shared legend. This keeps
    # method labels, colors, and ordering identical across every panel.
    panel_count = 3 if time_avg_keys else 2
    fig, axes = plt.subplots(1, panel_count, figsize=(6 * panel_count, 5))
    axes = np.atleast_1d(axes)

    ax = axes[0]
    for method, label, color in method_specs:
        y = [row.get(f"{method}_FWER", float("nan")) for row in fwer_rows]
        ax.plot(x_fwer, y, marker="o", linewidth=2, color=color, label=label)
    ax.set_xlabel("Number of false hypotheses")
    ax.set_ylabel("FWER")
    ax.set_title("FWER")
    ax.set_xticks(x_fwer)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for method, label, color in method_specs:
        y = [row.get(f"{method}_Power_avg", float("nan")) for row in power_rows]
        ax.plot(x_power, y, marker="o", linewidth=2, color=color, label=label)
    ax.set_xlabel("Number of false hypotheses")
    ax.set_ylabel("Average power")
    ax.set_title("Power")
    ax.set_xticks(x_power)
    ax.grid(True, alpha=0.3)

    if time_avg_keys:
        time_rows = sorted(results, key=lambda row: row["n_false"])
        x_time = [row["n_false"] for row in time_rows]
        ax = axes[2]
        for method, label, color in method_specs:
            y = [row.get(f"{method}_Time_avg_seconds", float("nan")) for row in time_rows]
            ax.plot(x_time, y, marker="o", linewidth=2, color=color, label=label)
        ax.set_xlabel("Number of false hypotheses")
        ax.set_ylabel("Average CPU time per dataset (seconds)")
        ax.set_title("CPU time")
        ax.set_xticks(x_time)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=min(5, len(labels)),
        frameon=True,
    )
    fig.suptitle("CMS Procedure Comparison", fontsize=16)
    fig.tight_layout(rect=(0.0, 0.16, 1.0, 0.94))
    combined_path = f"{output_prefix}_combined.png"
    fig.savefig(combined_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(combined_path)

    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate CMS bottom-up procedure: thresholds, FWER, and power."
    )
    parser.add_argument("--K", type=int, default=10, help="Number of hypotheses.")
    parser.add_argument("--alpha", type=float, default=0.05, help="FWER target.")
    parser.add_argument(
        "--target-power",
        type=float,
        default=0.9,
        help="Used for theta = Phi^{-1}(alpha/K) - Phi^{-1}(target_power).",
    )
    parser.add_argument(
        "--objective",
        type=str,
        choices=["bayes", "pi1"],
        default="bayes",
        help="Choose h(u)=1+g_theta(u) for bayes, or h(u)=g_theta(u) for pi1.",
    )
    parser.add_argument(
        "--B-threshold",
        type=int,
        default=20000,
        help="Monte Carlo samples per k for threshold estimation (Algorithm 2).",
    )
    parser.add_argument(
        "--n-sim",
        type=int,
        default=5000,
        help="Number of simulation replicates per truth configuration.",
    )
    parser.add_argument(
        "--mu-alt",
        type=float,
        default=2.5,
        help="Mean shift for false hypotheses in Z ~ N(mu_i,1).",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=0.0,
        help="Common pairwise correlation for equicorrelated normal statistics.",
    )
    parser.add_argument("--seed", type=int, default=123, help="RNG seed.")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of worker processes for threshold estimation and simulation (-1=all CPUs).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2000,
        help="Chunk size for threshold estimation.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="cms_normal_results.json",
        help="Path for JSON output.",
    )
    parser.add_argument(
        "--plot-prefix",
        type=str,
        default="cms_normal_results",
        help="Prefix for output plot files; set empty string to skip plotting.",
    )
    args = parser.parse_args()
    if args.n_jobs != 1 and (Parallel is None or delayed is None):
        print("joblib not found; falling back to n_jobs=1.")
        args.n_jobs = 1
    K = args.K
    alpha = args.alpha
    theta = float(normal_ppf(np.array([alpha / K]))[0] - normal_ppf(np.array([args.target_power]))[0])

    def f(u):
        return g_theta_normal_p(u, theta)

    if args.objective == "bayes":
        def h(u):
            return 1.0 + g_theta_normal_p(u, theta)
    else:
        def h(u):
            return 1.0

    print(f"Estimating thresholds t2..t{K} (objective={args.objective})...")
    thresholds = estimate_cms_thresholds(
        K=K,
        B=args.B_threshold,
        f=f,
        h=h,
        alpha=alpha,
        seed=args.seed,
        chunk_size=args.chunk_size,
        n_jobs=args.n_jobs,
        verbose=True,
    )
    print("Thresholds:", thresholds)

    rng = np.random.default_rng(args.seed + 1000)
    configs = build_default_configs(K)
    rows = []
    print("\nRunning power/FWER simulation...")
    for true_null in configs:
        n_false = int((~true_null).sum())
        mu = np.where(true_null, 0.0, args.mu_alt)
        stats = simulate_config(
            n_sim=args.n_sim,
            mu=mu,
            true_null=true_null,
            rho=args.rho,
            thresholds_t2_to_tk=thresholds,
            f=f,
            h=h,
            alpha=alpha,
            rng=rng,
            n_jobs=args.n_jobs,
        )
        row = {
            "K": K,
            "n_false": n_false,
            "n_true": int(true_null.sum()),
            "mu_alt": args.mu_alt,
            "rho": args.rho,
            **stats,
        }
        rows.append(row)
        print(
            f"n_false={n_false:2d} | "
            f"CMS(FWER={row['CMS_FWER']:.4f}, Power_any={row['CMS_Power_any']:.4f}, Power_avg={row['CMS_Power_avg']:.4f}) | "
            f"Holm(FWER={row['Holm_FWER']:.4f}, Power_any={row['Holm_Power_any']:.4f}, Power_avg={row['Holm_Power_avg']:.4f}) | "
            f"GTXR(FWER={row['GTXR_FWER']:.4f}, Power_any={row['GTXR_Power_any']:.4f}, Power_avg={row['GTXR_Power_avg']:.4f}) | "
            f"Hommel(FWER={row['Hommel_FWER']:.4f}, Power_any={row['Hommel_Power_any']:.4f}, Power_avg={row['Hommel_Power_avg']:.4f})"
        )

    payload = {
        "params": {
            "K": K,
            "alpha": alpha,
            "target_power": args.target_power,
            "theta": float(theta),
            "objective": args.objective,
            "B_threshold": args.B_threshold,
            "n_sim": args.n_sim,
            "mu_alt": args.mu_alt,
            "rho": args.rho,
            "seed": args.seed,
            "n_jobs": args.n_jobs,
        },
        "thresholds_t2_to_tK": thresholds.tolist(),
        "results": rows,
    }
    with open(args.output_json, "w", encoding="utf-8") as f_out:
        json.dump(payload, f_out, indent=2)

    print(f"\nSaved results to: {args.output_json}")
    if args.plot_prefix:
        plot_paths = plot_simulation_results(rows, args.plot_prefix)
        for plot_path in plot_paths:
            print(f"Saved plot to: {plot_path}")


if __name__ == "__main__":
    main()
