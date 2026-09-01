import importlib
import json
import os
from statistics import NormalDist

import numpy as np


sim_mod = importlib.import_module("simulate_cms_normal")
importlib.reload(sim_mod)


# Edit these settings, then run this file in IDLE with Run -> Run Module (F5).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
THRESHOLD_FILE = os.path.join(BASE_DIR, "cms_thresholds_K10.json")
OUTPUT_PREFIX = os.path.join(BASE_DIR, "runtime_by_dimension_pimix_0p7")

K_MIN = 2
K_MAX = 10
N_SIM = 10000
RHO = 0.0
SEED = 123
N_JOBS = -1

# Pi_mix places equal prior mass on null and alternative coordinates.
DATA_OBJECTIVE = "bayes"
DATA_TARGET_POWER = 0.7
ALTERNATIVE_PROBABILITY = 0.5


def _load_threshold_payload() -> dict:
    with open(THRESHOLD_FILE, "r", encoding="utf-8") as f_in:
        payload = json.load(f_in)

    saved_k = int(payload["params"]["K"])
    if saved_k < K_MAX:
        raise ValueError(
            f"Threshold file only supports K <= {saved_k}, but K_MAX={K_MAX}."
        )
    return payload


def _saved_theta(spec: dict, alpha: float, saved_k: int) -> float:
    if "theta" in spec:
        return float(spec["theta"])
    std = NormalDist()
    return std.inv_cdf(alpha / saved_k) - std.inv_cdf(
        float(spec["target_power"])
    )


def _build_bu_procedures(payload: dict, dimension: int) -> list[dict]:
    params = payload["params"]
    alpha = float(params["alpha"])
    saved_k = int(params["K"])
    alternative = params.get("alternative", "normal")
    beta_a = float(params.get("beta_a", 0.3))
    procedures = []

    for spec in payload["bu_procedures"]:
        objective = spec["objective"]
        target_power = float(spec["target_power"])
        all_thresholds = np.asarray(spec["thresholds_t2_to_tK"], dtype=float)
        thresholds = all_thresholds[: dimension - 1]
        if thresholds.size != dimension - 1:
            raise ValueError(
                f"{spec['name']} does not contain thresholds through t_{dimension}."
            )

        if alternative == "normal":
            theta = _saved_theta(spec, alpha, saved_k)

            def f(u, theta=theta):
                return sim_mod.g_theta_normal_p(u, theta)

            if objective == "bayes":
                def h(u, theta=theta):
                    return 1.0 + sim_mod.g_theta_normal_p(u, theta)
            elif objective == "pi1":
                def h(u):
                    return 1.0
            else:
                raise ValueError(f"Unknown BU objective: {objective}")
        elif alternative == "beta_a1":
            def f(u, beta_a=beta_a):
                return sim_mod.beta_a1_density(u, beta_a)

            if objective == "bayes":
                def h(u, beta_a=beta_a):
                    return 1.0 + sim_mod.beta_a1_density(u, beta_a)
            elif objective == "pi1":
                def h(u):
                    return 1.0
            else:
                raise ValueError(f"Unknown BU objective: {objective}")
        else:
            raise ValueError("Threshold alternative must be 'normal' or 'beta_a1'.")

        procedures.append(
            {
                "name": spec["name"],
                "thresholds": thresholds,
                "f": f,
                "h": h,
                "objective": objective,
                "target_power": target_power,
            }
        )

    return procedures


def _find_data_spec(payload: dict) -> dict:
    for spec in payload["bu_procedures"]:
        if (
            spec["objective"] == DATA_OBJECTIVE
            and np.isclose(float(spec["target_power"]), DATA_TARGET_POWER)
        ):
            return spec
    raise ValueError(
        f"No BU {DATA_OBJECTIVE} procedure with target power "
        f"{DATA_TARGET_POWER} exists in the threshold file."
    )


def _dimension_data_theta(payload: dict, dimension: int) -> float:
    data_spec = _find_data_spec(payload)
    alpha = float(payload["params"]["alpha"])
    target_power = float(data_spec["target_power"])
    std = NormalDist()
    return std.inv_cdf(alpha / dimension) - std.inv_cdf(target_power)


def _generate_pimix_data(
    rng: np.random.Generator,
    n_sim: int,
    dimension: int,
    rho: float,
    payload: dict,
) -> tuple[np.ndarray, np.ndarray]:
    params = payload["params"]
    alternative = params.get("alternative", "normal")
    rho_min = -1.0 / (dimension - 1)
    if not (rho_min <= rho < 1.0):
        raise ValueError(f"For K={dimension}, rho must be in [{rho_min}, 1).")

    if rho == 0.0:
        latent = rng.standard_normal((n_sim, dimension))
    else:
        covariance = np.full((dimension, dimension), rho, dtype=float)
        np.fill_diagonal(covariance, 1.0)
        latent = rng.multivariate_normal(
            mean=np.zeros(dimension), cov=covariance, size=n_sim
        )

    is_alternative = rng.random((n_sim, dimension)) < ALTERNATIVE_PROBABILITY
    uniform = sim_mod.normal_cdf(latent)

    if alternative == "normal":
        theta = _dimension_data_theta(payload, dimension)
        z = latent + theta * is_alternative
        return sim_mod.normal_cdf(z), is_alternative

    if alternative == "beta_a1":
        beta_a = float(params.get("beta_a", 0.3))
        p_values = uniform.copy()
        p_values[is_alternative] = sim_mod.beta_a1_ppf(
            uniform[is_alternative], beta_a
        )
        return p_values, is_alternative

    raise ValueError("Threshold alternative must be 'normal' or 'beta_a1'.")


def _evaluate_mixture_one_p(
    p: np.ndarray,
    is_alternative: np.ndarray,
    alpha: float,
    procedures: list[dict],
) -> dict:
    true_null = ~is_alternative
    false_mask = is_alternative
    n_false = int(false_mask.sum())
    return sim_mod.evaluate_methods_one_p(
        p,
        true_null,
        false_mask,
        n_false,
        alpha,
        procedures,
    )


def _benchmark_dimension(
    payload: dict,
    dimension: int,
    p_values: np.ndarray,
    is_alternative: np.ndarray,
) -> dict:
    alpha = float(payload["params"]["alpha"])
    procedures = _build_bu_procedures(payload, dimension)
    row = {"K": dimension}
    method_names = [procedure["name"] for procedure in procedures] + [
        "GTXR",
        "Holm",
        "Hommel",
    ]
    totals = {
        name: {"fwer_events": 0, "true_rejections": 0.0, "runtime": 0.0}
        for name in method_names
    }
    alternative_count = int(is_alternative.sum())

    # Use the same sample-major evaluator and method order as the fixed-K study.
    use_parallel = (
        N_JOBS != 1
        and sim_mod.Parallel is not None
        and sim_mod.delayed is not None
    )
    if use_parallel:
        all_method_outputs = sim_mod.Parallel(
            n_jobs=N_JOBS, prefer="processes"
        )(
            sim_mod.delayed(_evaluate_mixture_one_p)(
                p_values[index],
                is_alternative[index],
                alpha,
                procedures,
            )
            for index in range(p_values.shape[0])
        )
    else:
        all_method_outputs = [
            _evaluate_mixture_one_p(
                p_values[index],
                is_alternative[index],
                alpha,
                procedures,
            )
            for index in range(p_values.shape[0])
        ]

    for index, method_outputs in enumerate(all_method_outputs):
        n_false = int(is_alternative[index].sum())
        for name, method_output in method_outputs.items():
            any_true, _, tpr = method_output["summary"]
            totals[name]["fwer_events"] += int(any_true)
            if n_false > 0:
                totals[name]["true_rejections"] += float(tpr) * n_false
            totals[name]["runtime"] += float(method_output["time_seconds"])

    for name in method_names:
        runtime = totals[name]["runtime"] / p_values.shape[0]
        fwer = totals[name]["fwer_events"] / p_values.shape[0]
        if alternative_count > 0:
            tpr = totals[name]["true_rejections"] / alternative_count
        else:
            tpr = float("nan")
        row[f"{name}_Time_avg_seconds"] = float(runtime)
        row[f"{name}_FWER"] = float(fwer)
        row[f"{name}_TPR"] = float(tpr)
        print(
            f"  {name}: runtime={runtime:.8g}, FWER={fwer:.5f}, TPR={tpr:.5f}"
        )

    return row


def _method_styles(method_names: list[str]) -> dict[str, dict]:
    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
    ]
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h"]
    return {
        method: {
            "color": colors[index % len(colors)],
            "marker": markers[index % len(markers)],
            "label": sim_mod._format_method_label(method),
        }
        for index, method in enumerate(method_names)
    }


def _draw_metric(
    ax,
    results: list[dict],
    method_names: list[str],
    styles: dict[str, dict],
    key_suffix: str,
    ylabel: str,
    title: str,
) -> None:
    dimensions = [row["K"] for row in results]
    for method in method_names:
        style = styles[method]
        values = [row[f"{method}_{key_suffix}"] for row in results]
        ax.plot(
            dimensions,
            values,
            color=style["color"],
            marker=style["marker"],
            linewidth=2,
            markersize=7,
            label=style["label"],
        )
    ax.set_xlabel("Dimension K")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(dimensions)
    ax.grid(True, alpha=0.3)


def _plot_results(
    results: list[dict],
    method_names: list[str],
    alpha: float,
) -> list[str]:
    if sim_mod.plt is None:
        raise RuntimeError("matplotlib is required to generate plots.")

    styles = _method_styles(method_names)
    plot_specs = [
        (
            "Time_avg_seconds",
            "Average CPU time per dataset (seconds)",
            "CPU time",
            f"{OUTPUT_PREFIX}.png",
        ),
        ("FWER", "FWER", "Family-wise error rate", f"{OUTPUT_PREFIX}_fwer.png"),
        (
            "TPR",
            "Power (TPR)",
            "True positive rate",
            f"{OUTPUT_PREFIX}_power_tpr.png",
        ),
    ]
    output_paths = []

    for key_suffix, ylabel, title, output_path in plot_specs:
        fig, ax = sim_mod.plt.subplots(figsize=(10, 6))
        _draw_metric(
            ax, results, method_names, styles, key_suffix, ylabel, title
        )
        if key_suffix == "FWER":
            ax.axhline(alpha, color="black", linestyle="--", linewidth=1.5)
            ax.set_ylim(bottom=0.0)
        elif key_suffix == "TPR":
            ax.set_ylim(0.0, 1.0)
        else:
            ax.set_ylim(bottom=0.0)
        ax.legend(
            loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0
        )
        fig.tight_layout(rect=(0.0, 0.0, 0.78, 1.0))
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        sim_mod.plt.close(fig)
        output_paths.append(output_path)

    fig, axes = sim_mod.plt.subplots(1, 3, figsize=(18, 5))
    for ax, (key_suffix, ylabel, title, _) in zip(axes, plot_specs):
        _draw_metric(
            ax, results, method_names, styles, key_suffix, ylabel, title
        )
        if key_suffix == "FWER":
            ax.axhline(alpha, color="black", linestyle="--", linewidth=1.5)
            ax.set_ylim(bottom=0.0)
        elif key_suffix == "TPR":
            ax.set_ylim(0.0, 1.0)
        else:
            ax.set_ylim(bottom=0.0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=min(5, len(labels)),
        frameon=True,
    )
    fig.suptitle(
        r"Performance under true BU $\Pi_{\mathrm{mix}}(0.7)$ data generation",
        fontsize=16,
    )
    fig.tight_layout(rect=(0.0, 0.17, 1.0, 0.93))
    combined_path = f"{OUTPUT_PREFIX}_combined.png"
    fig.savefig(combined_path, dpi=150, bbox_inches="tight")
    sim_mod.plt.close(fig)
    output_paths.append(combined_path)
    return output_paths


def main() -> None:
    payload = _load_threshold_payload()
    rng = np.random.default_rng(SEED)
    results = []

    if N_JOBS != 1 and (
        sim_mod.Parallel is None or sim_mod.delayed is None
    ):
        print("joblib not found; falling back to N_JOBS=1 timing.")

    for dimension in range(K_MIN, K_MAX + 1):
        if payload["params"].get("alternative", "normal") == "normal":
            data_theta = _dimension_data_theta(payload, dimension)
            print(
                f"Benchmarking K={dimension} with {N_SIM} datasets "
                f"(theta_K={data_theta:.8g})..."
            )
        else:
            data_theta = None
            print(f"Benchmarking K={dimension} with {N_SIM} datasets...")
        p_values, is_alternative = _generate_pimix_data(
            rng=rng,
            n_sim=N_SIM,
            dimension=dimension,
            rho=RHO,
            payload=payload,
        )
        row = _benchmark_dimension(
            payload, dimension, p_values, is_alternative
        )
        if data_theta is not None:
            row["data_theta"] = float(data_theta)
        results.append(row)

    method_names = [spec["name"] for spec in payload["bu_procedures"]]
    method_names.extend(["Holm", "Hommel", "GTXR"])

    output = {
        "threshold_file": THRESHOLD_FILE,
        "settings": {
            "K_min": K_MIN,
            "K_max": K_MAX,
            "n_sim": N_SIM,
            "rho": RHO,
            "seed": SEED,
            "n_jobs": N_JOBS,
            "data_objective": DATA_OBJECTIVE,
            "data_target_power": DATA_TARGET_POWER,
            "alternative_probability": ALTERNATIVE_PROBABILITY,
            "data_generation": "Pi_mix: independent 0.5 null/alternative labels",
            "normal_theta_formula": "Phi^-1(alpha / K) - Phi^-1(target_power)",
            "timing_clock": "time.process_time",
            "timing_definition": "mean CPU execution time per procedure call",
            "fwer_definition": "P(at least one rejected true null)",
            "tpr_definition": "total rejected alternatives / total alternatives",
        },
        "results": results,
    }
    json_path = f"{OUTPUT_PREFIX}.json"
    with open(json_path, "w", encoding="utf-8") as f_out:
        json.dump(output, f_out, indent=2)

    plot_paths = _plot_results(
        results, method_names, float(payload["params"]["alpha"])
    )
    print(f"Saved benchmark results to: {json_path}")
    for plot_path in plot_paths:
        print(f"Saved plot to: {plot_path}")


if __name__ == "__main__":
    main()
