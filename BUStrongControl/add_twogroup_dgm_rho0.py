import importlib
import json
import os
from statistics import NormalDist

import numpy as np


sim_mod = importlib.import_module("simulate_cms_normal")
importlib.reload(sim_mod)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Only these rho=0 studies receive the additional two-group DGM point.
STUDIES = [
    {"folder": "K10dg03", "target_power": 0.3},
    {"folder": "K5dg03", "target_power": 0.3},
    {"folder": "K10dg07", "target_power": 0.7},
    {"folder": "K5dg07", "target_power": 0.7},
]

ALTERNATIVE_PROBABILITY = 0.5
N_SIM_OVERRIDE = None  # None uses n_sim from each source JSON.
N_JOBS_OVERRIDE = None  # None uses n_jobs from each source JSON.
SEED_OFFSET = 2000
CHUNK_SIZE = 2000
FORCE_RECOMPUTE = False

DISPLAY_METHODS = [
    "BU_bayes_tp_0p3",
    "BU_bayes_tp_0p7",
    "BU_pi1_tp_0p3",
    "Hommel",
    "GTXR",
]

DISPLAY_STYLES = {
    "BU_bayes_tp_0p3": {"color": "red", "marker": "o"},
    "BU_bayes_tp_0p7": {"color": "green", "marker": "s"},
    "BU_pi1_tp_0p3": {"color": "blue", "marker": "^"},
    "Hommel": {"color": "cyan", "marker": "D"},
    "GTXR": {"color": "black", "marker": "*"},
}


def _build_bu_procedures(payload: dict) -> list[dict]:
    params = payload["threshold_params"]
    alpha = float(params["alpha"])
    K = int(params["K"])
    alternative = params.get("alternative", "normal")
    if alternative != "normal":
        raise ValueError("This two-group DGM script currently requires normal thresholds.")

    std = NormalDist()
    procedures = []
    for spec in payload["bu_procedures"]:
        objective = spec["objective"]
        target_power = float(spec["target_power"])
        theta = float(
            spec.get(
                "theta",
                std.inv_cdf(alpha / K) - std.inv_cdf(target_power),
            )
        )

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

        thresholds = np.asarray(spec["thresholds_t2_to_tK"], dtype=float)
        if thresholds.size != K - 1:
            raise ValueError(
                f"{spec['name']} has {thresholds.size} thresholds; expected {K - 1}."
            )
        procedures.append(
            {
                "name": spec["name"],
                "objective": objective,
                "target_power": target_power,
                "thresholds": thresholds,
                "f": f,
                "h": h,
            }
        )
    return procedures


def _evaluate_one(
    p_values: np.ndarray,
    is_alternative: np.ndarray,
    alpha: float,
    procedures: list[dict],
) -> tuple[int, dict]:
    false_mask = np.asarray(is_alternative, dtype=bool)
    true_null = ~false_mask
    n_false = int(false_mask.sum())
    output = sim_mod.evaluate_methods_one_p(
        p_values,
        true_null,
        false_mask,
        n_false,
        alpha,
        procedures,
    )
    return n_false, output


def _simulate_two_group(
    payload: dict,
    data_target_power: float,
) -> dict:
    params = payload["threshold_params"]
    simulation_params = payload["simulation_params"]
    K = int(params["K"])
    alpha = float(params["alpha"])
    rho = float(simulation_params["rho"])
    if not np.isclose(rho, 0.0):
        raise ValueError(f"Expected rho=0, found rho={rho}.")

    n_sim = int(
        simulation_params["n_sim"]
        if N_SIM_OVERRIDE is None
        else N_SIM_OVERRIDE
    )
    n_jobs = int(
        simulation_params.get("n_jobs", 1)
        if N_JOBS_OVERRIDE is None
        else N_JOBS_OVERRIDE
    )
    seed = int(simulation_params.get("seed", 123)) + SEED_OFFSET
    theta = float(simulation_params["mu_alt"])
    expected_theta = NormalDist().inv_cdf(alpha / K) - NormalDist().inv_cdf(
        data_target_power
    )
    if not np.isclose(theta, expected_theta):
        raise ValueError(
            f"Saved mu_alt={theta} does not match target power "
            f"{data_target_power} for K={K}: expected {expected_theta}."
        )

    procedures = _build_bu_procedures(payload)
    method_names = [procedure["name"] for procedure in procedures] + [
        "GTXR",
        "Holm",
        "Hommel",
    ]
    totals = {
        name: {
            "fwer_events": 0,
            "any_power_events": 0,
            "tpr_sum": 0.0,
            "power_datasets": 0,
            "runtime": 0.0,
        }
        for name in method_names
    }
    n_false_histogram = np.zeros(K + 1, dtype=np.int64)

    use_parallel = (
        n_jobs != 1
        and sim_mod.Parallel is not None
        and sim_mod.delayed is not None
    )
    if n_jobs != 1 and not use_parallel:
        print("joblib not found; falling back to n_jobs=1.")

    rng = np.random.default_rng(seed)
    completed = 0
    while completed < n_sim:
        chunk_n = min(CHUNK_SIZE, n_sim - completed)
        is_alternative = (
            rng.random((chunk_n, K)) < ALTERNATIVE_PROBABILITY
        )
        z = rng.standard_normal((chunk_n, K))
        z += theta * is_alternative
        p_values = sim_mod.normal_cdf(z)

        if use_parallel:
            chunk_outputs = sim_mod.Parallel(
                n_jobs=n_jobs,
                prefer="processes",
            )(
                sim_mod.delayed(_evaluate_one)(
                    p_values[index],
                    is_alternative[index],
                    alpha,
                    procedures,
                )
                for index in range(chunk_n)
            )
        else:
            chunk_outputs = [
                _evaluate_one(
                    p_values[index],
                    is_alternative[index],
                    alpha,
                    procedures,
                )
                for index in range(chunk_n)
            ]

        for n_false, method_outputs in chunk_outputs:
            n_false_histogram[n_false] += 1
            for name, method_output in method_outputs.items():
                any_true, any_false, tpr = method_output["summary"]
                totals[name]["fwer_events"] += int(any_true)
                totals[name]["runtime"] += float(method_output["time_seconds"])
                if n_false > 0:
                    totals[name]["any_power_events"] += int(any_false)
                    totals[name]["tpr_sum"] += float(tpr)
                    totals[name]["power_datasets"] += 1

        completed += chunk_n
        print(f"  completed {completed}/{n_sim} two-group datasets")

    results = {}
    for name in method_names:
        method = totals[name]
        power_datasets = int(method["power_datasets"])
        results[f"{name}_FWER"] = float(method["fwer_events"] / n_sim)
        results[f"{name}_Power_any"] = (
            float(method["any_power_events"] / power_datasets)
            if power_datasets > 0
            else float("nan")
        )
        results[f"{name}_Power_avg"] = (
            float(method["tpr_sum"] / power_datasets)
            if power_datasets > 0
            else float("nan")
        )
        results[f"{name}_Time_avg_seconds"] = float(
            method["runtime"] / n_sim
        )

    return {
        "model": {
            "name": "two_group",
            "K": K,
            "alpha": alpha,
            "rho": 0.0,
            "alternative_probability": ALTERNATIVE_PROBABILITY,
            "data_target_power": data_target_power,
            "theta": theta,
            "n_sim": n_sim,
            "seed": seed,
            "n_jobs": n_jobs,
            "null_pvalue_distribution": "Uniform(0,1)",
            "alternative_pvalue_density": "g_theta",
            "power_definition": "mean within-dataset TPR conditional on at least one alternative",
            "fwer_definition": "unconditional P(at least one rejected true null)",
        },
        "n_false_histogram": n_false_histogram.tolist(),
        "results": results,
    }


def _method_styles(methods: list[str]) -> dict[str, dict]:
    return {
        method: {
            **DISPLAY_STYLES[method],
            "label": sim_mod._format_method_label(method),
        }
        for method in methods
    }


def _plot_augmented_results(
    payload: dict,
    two_group: dict,
    source_json: str,
    data_target_power: float,
) -> list[str]:
    if sim_mod.plt is None:
        raise RuntimeError("matplotlib is required to generate plots.")

    rows = payload["results"]
    K = int(payload["threshold_params"]["K"])
    alpha = float(payload["threshold_params"]["alpha"])
    available_methods = [
        key[:-5]
        for key in rows[0]
        if key.endswith("_FWER")
    ]
    missing_methods = [
        method for method in DISPLAY_METHODS if method not in available_methods
    ]
    if missing_methods:
        raise ValueError(
            "Source JSON is missing requested plot methods: "
            + ", ".join(missing_methods)
        )
    methods = DISPLAY_METHODS
    styles = _method_styles(methods)
    two_group_label = f"mix ({data_target_power:g})"
    prefix = os.path.splitext(source_json)[0]
    outputs = []

    plot_specs = [
        (
            "FWER",
            [row for row in rows if int(row["n_false"]) <= K - 1],
            "FWER",
            "FWER by Configuration",
            f"{prefix}_twogroup_dgm_fwer.png",
        ),
        (
            "Power_avg",
            [row for row in rows if int(row["n_false"]) >= 1],
            "Average power",
            "Power by Configuration",
            f"{prefix}_twogroup_dgm_power.png",
        ),
    ]

    for metric, metric_rows, ylabel, title, output_path in plot_specs:
        metric_rows = sorted(metric_rows, key=lambda row: int(row["n_false"]))
        x = [int(row["n_false"]) for row in metric_rows]
        two_group_x = max(x) + 1
        fig, ax = sim_mod.plt.subplots(figsize=(7, 7))
        plotted_values = []
        for method in methods:
            style = styles[method]
            y = [float(row[f"{method}_{metric}"]) for row in metric_rows]
            two_group_value = float(
                two_group["results"][f"{method}_{metric}"]
            )
            plotted_values.extend(y)
            plotted_values.append(two_group_value)
            ax.plot(
                x,
                y,
                linewidth=2,
                markersize=5,
                marker=style["marker"],
                color=style["color"],
                label=style["label"],
            )
            ax.scatter(
                [two_group_x],
                [two_group_value],
                marker=style["marker"],
                s=70,
                color=style["color"],
                zorder=4,
            )

        if metric == "FWER":
            ax.axhline(
                alpha,
                color="black",
                linestyle="--",
                linewidth=1.3,
                label="_nolegend_",
            )
            plotted_values.append(alpha)
            finite_values = np.asarray(plotted_values, dtype=float)
            finite_values = finite_values[np.isfinite(finite_values)]
            if finite_values.size > 0:
                data_min = float(finite_values.min())
                data_max = float(finite_values.max())
                padding = max(0.001, 0.08 * (data_max - data_min))
                lower = max(0.0, data_min - padding)
                upper = min(1.0, data_max + padding)
                if np.isclose(lower, upper):
                    lower = max(0.0, lower - 0.001)
                    upper = min(1.0, upper + 0.001)
                ax.set_ylim(lower, upper)
        else:
            ax.axhline(
                data_target_power,
                color="0.45",
                linestyle=":",
                linewidth=1.3,
                label="Bonferroni",
            )
            plotted_values.append(data_target_power)
            finite_values = np.asarray(plotted_values, dtype=float)
            finite_values = finite_values[np.isfinite(finite_values)]
            if finite_values.size > 0:
                data_min = float(finite_values.min())
                data_max = float(finite_values.max())
                padding = max(0.02, 0.08 * (data_max - data_min))
                lower = max(0.0, data_min - padding)
                upper = min(1.0, data_max + padding)
                if np.isclose(lower, upper):
                    lower = max(0.0, lower - 0.02)
                    upper = min(1.0, upper + 0.02)
                ax.set_ylim(lower, upper)

        ax.axvline(
            two_group_x - 0.5,
            color="0.55",
            linestyle=":",
            linewidth=1.2,
        )
        ax.set_xlabel("Fixed number of false hypotheses / data-generating model")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x + [two_group_x])
        ax.set_xticklabels([str(value) for value in x] + [two_group_label])
        ax.grid(True, alpha=0.3)
        legend_location = "lower left" if metric == "FWER" else "upper left"
        ax.legend(loc=legend_location, frameon=True)
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        sim_mod.plt.close(fig)
        outputs.append(output_path)
    return outputs


def process_study(folder: str, data_target_power: float) -> list[str]:
    source_json = os.path.join(
        BASE_DIR,
        folder,
        "cms_compare_rho_0p0.json",
    )
    summary_json = os.path.join(
        BASE_DIR,
        folder,
        "cms_compare_rho_0p0_twogroup_dgm.json",
    )
    with open(source_json, "r", encoding="utf-8") as f_in:
        payload = json.load(f_in)

    if os.path.exists(summary_json) and not FORCE_RECOMPUTE:
        print(f"Using saved two-group result: {summary_json}")
        with open(summary_json, "r", encoding="utf-8") as f_in:
            two_group = json.load(f_in)
    else:
        print(
            f"Simulating {folder}: rho=0, target power={data_target_power}, "
            f"P(h_i=1)={ALTERNATIVE_PROBABILITY}"
        )
        two_group = _simulate_two_group(payload, data_target_power)
        with open(summary_json, "w", encoding="utf-8") as f_out:
            json.dump(two_group, f_out, indent=2)
        print(f"Saved two-group result: {summary_json}")

    return _plot_augmented_results(
        payload,
        two_group,
        source_json,
        data_target_power,
    )


def main() -> None:
    for study in STUDIES:
        paths = process_study(
            study["folder"],
            float(study["target_power"]),
        )
        for path in paths:
            print(f"Saved plot: {path}")


if __name__ == "__main__":
    main()
