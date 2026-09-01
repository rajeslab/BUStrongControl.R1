"""Simulate and plot improved-Hommel power minus standard Hommel power."""

import importlib
import json
import os
from statistics import NormalDist

import numpy as np


ih_mod = importlib.import_module("improved_hommel")
importlib.reload(ih_mod)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
THRESHOLD_FILE = os.path.join(
    BASE_DIR,
    "improved_hommel_thresholds_K10.json",
)

K = 10
ALPHA = 0.05
DATA_TARGET_POWER = 0.3
ALTERNATIVE_PROBABILITY = 0.5
N_SIM = 200_000
CHUNK_SIZE = 20_000
SEED = 123

METHOD_NAMES = [
    "Improved_Hommel_bayes_tp_0p3",
    "Improved_Hommel_bayes_tp_0p7",
    "Improved_Hommel_pi1_tp_0p3",
]

METHOD_STYLES = {
    "Improved_Hommel_bayes_tp_0p3": {
        "color": "#ff4b4b",
        "marker": "o",
        "label": r"IH $\Pi_{\mathrm{mix}}(0.3)-Hom$",
        "absolute_label": r"IH $\Pi_{\mathrm{mix}}(0.3)$",
        "mix_offset": -0.16,
    },
    "Improved_Hommel_bayes_tp_0p7": {
        "color": "#168a2b",
        "marker": "o",
        "label": r"IH $\Pi_{\mathrm{mix}}(0.7)-Hom$",
        "absolute_label": r"IH $\Pi_{\mathrm{mix}}(0.7)$",
        "mix_offset": 0.0,
    },
    "Improved_Hommel_pi1_tp_0p3": {
        "color": "#304ffe",
        "marker": "o",
        "label": r"IH $\Pi_1(0.3)-Hom$",
        "absolute_label": r"IH $\Pi_1(0.3)$",
        "mix_offset": 0.16,
    },
}

OUTPUT_PREFIX = os.path.join(
    BASE_DIR,
    "improved_hommel_power_gain_K10_tp_0p3",
)


def _target_token(target_power: float) -> str:
    return str(float(target_power)).replace(".", "p")


def _load_method_specs() -> tuple[dict, list[dict]]:
    with open(THRESHOLD_FILE, "r", encoding="utf-8") as f_in:
        payload = json.load(f_in)

    params = payload["params"]
    if int(params["K"]) != K:
        raise ValueError(
            f"Threshold file contains K={params['K']}; expected K={K}."
        )
    if not np.isclose(float(params["alpha"]), ALPHA):
        raise ValueError("Threshold alpha does not match ALPHA.")

    by_name = {spec["name"]: spec for spec in payload["procedures"]}
    missing = [name for name in METHOD_NAMES if name not in by_name]
    if missing:
        raise ValueError(
            "Threshold file is missing procedures: " + ", ".join(missing)
        )
    return payload, [by_name[name] for name in METHOD_NAMES]


def _chunk_metrics(
    p_values: np.ndarray,
    is_alternative: np.ndarray,
    method_specs: list[dict],
) -> tuple[dict[str, dict[str, float]], int]:
    """Return TPR sums and FWER counts for a common simulation chunk."""
    order = np.argsort(p_values, axis=1)
    p_sorted = np.take_along_axis(p_values, order, axis=1)
    alternative_sorted = np.take_along_axis(is_alternative, order, axis=1)
    true_null_sorted = ~alternative_sorted
    n_false = alternative_sorted.sum(axis=1)
    valid = n_false > 0
    n_valid = int(valid.sum())

    hommel = ih_mod._hommel_rejections_sorted_batch(p_sorted, ALPHA)
    candidates = ih_mod.improved_hommel_candidates_sorted_batch(
        p_sorted,
        ALPHA,
    )

    def summarize(decisions: np.ndarray) -> dict[str, float]:
        tpr = np.zeros(p_sorted.shape[0], dtype=float)
        tpr[valid] = (
            (decisions[valid] & alternative_sorted[valid]).sum(axis=1)
            / n_false[valid]
        )
        return {
            "tpr_sum": float(tpr[valid].sum()),
            "fwer_events": float(
                np.any(decisions & true_null_sorted, axis=1).sum()
            ),
        }

    metrics = {"Hommel": summarize(hommel)}
    for spec in method_specs:
        scores = ih_mod.improved_hommel_scores_sorted_batch(
            p_sorted,
            candidates,
            theta=float(spec["theta"]),
            objective=spec["objective"],
        )
        decisions = candidates & (scores > float(spec["threshold"]))[:, None]
        metrics[spec["name"]] = summarize(decisions)
    return metrics, n_valid


def _simulate_fixed_configuration(
    n_false: int,
    theta_data: float,
    method_specs: list[dict],
) -> dict:
    rng = np.random.default_rng(SEED + 100_003 * n_false)
    method_names = ["Hommel"] + [spec["name"] for spec in method_specs]
    totals = {
        name: {"tpr_sum": 0.0, "fwer_events": 0.0}
        for name in method_names
    }
    n_valid_total = 0

    completed = 0
    while completed < N_SIM:
        n_chunk = min(CHUNK_SIZE, N_SIM - completed)
        is_alternative = np.zeros((n_chunk, K), dtype=bool)
        is_alternative[:, :n_false] = True
        z_values = rng.standard_normal((n_chunk, K))
        z_values[:, :n_false] += theta_data
        p_values = ih_mod.ndtr(z_values)

        metrics, n_valid = _chunk_metrics(
            p_values,
            is_alternative,
            method_specs,
        )
        for name, values in metrics.items():
            totals[name]["tpr_sum"] += values["tpr_sum"]
            totals[name]["fwer_events"] += values["fwer_events"]
        n_valid_total += n_valid
        completed += n_chunk

    hommel_power = (
        totals["Hommel"]["tpr_sum"] / n_valid_total
        if n_valid_total > 0
        else None
    )
    row = {
        "n_false": n_false,
        "Hommel_FWER": totals["Hommel"]["fwer_events"] / N_SIM,
        "Hommel_Power_avg": hommel_power,
    }
    for spec in method_specs:
        name = spec["name"]
        power = (
            totals[name]["tpr_sum"] / n_valid_total
            if n_valid_total > 0
            else None
        )
        row[f"{name}_FWER"] = totals[name]["fwer_events"] / N_SIM
        row[f"{name}_Power_avg"] = power
        row[f"{name}_Minus_Hommel"] = (
            power - hommel_power if power is not None else None
        )
    return row


def _simulate_two_group(
    theta_data: float,
    method_specs: list[dict],
) -> dict:
    rng = np.random.default_rng(SEED + 2_000)
    method_names = ["Hommel"] + [spec["name"] for spec in method_specs]
    totals = {
        name: {"tpr_sum": 0.0, "fwer_events": 0.0}
        for name in method_names
    }
    n_valid_total = 0
    n_false_histogram = np.zeros(K + 1, dtype=np.int64)

    completed = 0
    while completed < N_SIM:
        n_chunk = min(CHUNK_SIZE, N_SIM - completed)
        is_alternative = (
            rng.random((n_chunk, K)) < ALTERNATIVE_PROBABILITY
        )
        n_false_histogram += np.bincount(
            is_alternative.sum(axis=1),
            minlength=K + 1,
        )
        z_values = rng.standard_normal((n_chunk, K))
        z_values += theta_data * is_alternative
        p_values = ih_mod.ndtr(z_values)

        metrics, n_valid = _chunk_metrics(
            p_values,
            is_alternative,
            method_specs,
        )
        for name, values in metrics.items():
            totals[name]["tpr_sum"] += values["tpr_sum"]
            totals[name]["fwer_events"] += values["fwer_events"]
        n_valid_total += n_valid
        completed += n_chunk

    row = {
        "model": "two_group",
        "alternative_probability": ALTERNATIVE_PROBABILITY,
        "power_datasets": n_valid_total,
        "n_false_histogram": n_false_histogram.tolist(),
        "Hommel_FWER": totals["Hommel"]["fwer_events"] / N_SIM,
        "Hommel_Power_avg": totals["Hommel"]["tpr_sum"] / n_valid_total,
    }
    for spec in method_specs:
        name = spec["name"]
        power = totals[name]["tpr_sum"] / n_valid_total
        row[f"{name}_FWER"] = totals[name]["fwer_events"] / N_SIM
        row[f"{name}_Power_avg"] = power
        row[f"{name}_Minus_Hommel"] = power - row["Hommel_Power_avg"]
    return row


def _plot_results(
    fixed_results: list[dict],
    two_group_result: dict,
    method_specs: list[dict],
) -> str:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required to generate the plot.") from exc

    power_rows = [row for row in fixed_results if row["n_false"] >= 1]
    x_fixed = np.array([row["n_false"] for row in power_rows])
    mix_x = K + 1
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    all_differences = []

    for spec in method_specs:
        name = spec["name"]
        style = METHOD_STYLES[name]
        difference_key = f"{name}_Minus_Hommel"
        y_fixed = np.array([row[difference_key] for row in power_rows])
        y_mix = float(two_group_result[difference_key])
        mix_marker_x = mix_x + float(style["mix_offset"])
        all_differences.extend(y_fixed.tolist())
        all_differences.append(y_mix)

        ax.plot(
            x_fixed,
            y_fixed,
            color=style["color"],
            marker=style["marker"],
            markersize=4.5,
            linewidth=1.8,
            label=style["label"],
        )
        ax.scatter(
            [mix_marker_x],
            [y_mix],
            color=style["color"],
            marker=style["marker"],
            s=50,
            zorder=5,
        )

    # Mark configurations whose data generation agrees with the objective.
    token = _target_token(DATA_TARGET_POWER)
    matching_pi1 = f"Improved_Hommel_pi1_tp_{token}"
    matching_mix = f"Improved_Hommel_bayes_tp_{token}"
    spec_names = {spec["name"] for spec in method_specs}
    if matching_pi1 in spec_names:
        y_value = power_rows[0][f"{matching_pi1}_Minus_Hommel"]
        ax.scatter(
            [1],
            [y_value],
            s=130,
            facecolors="none",
            edgecolors="black",
            linewidths=1.7,
            zorder=7,
        )
    if matching_mix in spec_names:
        y_value = two_group_result[f"{matching_mix}_Minus_Hommel"]
        matching_style = METHOD_STYLES[matching_mix]
        ax.scatter(
            [mix_x + float(matching_style["mix_offset"])],
            [y_value],
            s=130,
            facecolors="none",
            edgecolors="black",
            linewidths=1.7,
            zorder=7,
        )

    ax.axhline(0.0, color="black", linewidth=1.2)
    ax.set_xticks(list(range(1, K + 1)) + [mix_x])
    ax.set_xticklabels([str(value) for value in range(1, K + 1)] + ["mix"])
    ax.set_xlim(0.5, mix_x + 0.5)
    finite = np.asarray(all_differences, dtype=float)
    finite = finite[np.isfinite(finite)]
    span = max(float(finite.max() - finite.min()), 0.001)
    ax.set_ylim(float(finite.min() - 0.12 * span), float(finite.max() + 0.12 * span))
    ax.set_xlabel("Number of Non-nulls")
    ax.set_ylabel("Improved Hommel (IH) - Hommel")
    ax.legend(loc="upper right", frameon=True)
    ax.grid(False)
    fig.tight_layout()

    output_path = f"{OUTPUT_PREFIX}.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _plot_absolute_metric(
    fixed_results: list[dict],
    two_group_result: dict,
    method_specs: list[dict],
    metric: str,
) -> str:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required to generate the plot.") from exc

    if metric == "Power_avg":
        rows = [row for row in fixed_results if row["n_false"] >= 1]
        ylabel = "Power (average TPR)"
        suffix = "power"
    elif metric == "FWER":
        rows = [row for row in fixed_results if row["n_false"] <= K - 1]
        ylabel = "FWER"
        suffix = "fwer"
    else:
        raise ValueError("metric must be 'Power_avg' or 'FWER'.")

    x_fixed = np.array([row["n_false"] for row in rows])
    mix_x = K + 1
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    plotted_values = []

    for spec in method_specs:
        name = spec["name"]
        style = METHOD_STYLES[name]
        metric_key = f"{name}_{metric}"
        y_fixed = np.array([row[metric_key] for row in rows], dtype=float)
        y_mix = float(two_group_result[metric_key])
        mix_marker_x = mix_x + float(style["mix_offset"])
        plotted_values.extend(y_fixed.tolist())
        plotted_values.append(y_mix)

        ax.plot(
            x_fixed,
            y_fixed,
            color=style["color"],
            marker=style["marker"],
            markersize=4.5,
            linewidth=1.8,
            label=style["absolute_label"],
        )
        ax.scatter(
            [mix_marker_x],
            [y_mix],
            color=style["color"],
            marker=style["marker"],
            s=50,
            zorder=5,
        )

    # Only the matching red mix point is circled at the mix location.
    matching_mix = f"Improved_Hommel_bayes_tp_{_target_token(DATA_TARGET_POWER)}"
    spec_names = {spec["name"] for spec in method_specs}
    if matching_mix in spec_names:
        style = METHOD_STYLES[matching_mix]
        ax.scatter(
            [mix_x + float(style["mix_offset"])],
            [two_group_result[f"{matching_mix}_{metric}"]],
            s=130,
            facecolors="none",
            edgecolors="black",
            linewidths=1.7,
            zorder=7,
        )

    if metric == "FWER":
        ax.axhline(
            ALPHA,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label=r"$\alpha=0.05$",
        )
        plotted_values.append(ALPHA)

    ax.set_xticks(x_fixed.tolist() + [mix_x])
    ax.set_xticklabels([str(value) for value in x_fixed] + ["mix"])
    ax.set_xlim(float(x_fixed.min() - 0.5), mix_x + 0.5)
    finite = np.asarray(plotted_values, dtype=float)
    finite = finite[np.isfinite(finite)]
    span = max(float(finite.max() - finite.min()), 0.001)
    lower = max(0.0, float(finite.min() - 0.10 * span))
    upper = min(1.0, float(finite.max() + 0.10 * span))
    ax.set_ylim(lower, upper)
    ax.set_xlabel("Number of Non-nulls")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", frameon=True)
    ax.grid(False)
    fig.tight_layout()

    output_path = f"{OUTPUT_PREFIX}_{suffix}.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    threshold_payload, method_specs = _load_method_specs()
    theta_data = NormalDist().inv_cdf(ALPHA / K) - NormalDist().inv_cdf(
        DATA_TARGET_POWER
    )

    fixed_results = []
    for n_false in range(0, K + 1):
        print(f"Simulating fixed configuration n_false={n_false}/{K}")
        fixed_results.append(
            _simulate_fixed_configuration(
                n_false,
                theta_data,
                method_specs,
            )
        )

    print("Simulating two-group mix configuration")
    two_group_result = _simulate_two_group(theta_data, method_specs)

    output = {
        "params": {
            "K": K,
            "alpha": ALPHA,
            "data_target_power": DATA_TARGET_POWER,
            "theta_data": theta_data,
            "alternative_probability": ALTERNATIVE_PROBABILITY,
            "n_sim": N_SIM,
            "chunk_size": CHUNK_SIZE,
            "seed": SEED,
            "threshold_file": THRESHOLD_FILE,
            "power_definition": (
                "mean within-dataset TPR conditional on at least one alternative"
            ),
        },
        "method_thresholds": method_specs,
        "fixed_results": fixed_results,
        "two_group_result": two_group_result,
        "threshold_params": threshold_payload["params"],
    }
    json_path = f"{OUTPUT_PREFIX}.json"
    with open(json_path, "w", encoding="utf-8") as f_out:
        json.dump(output, f_out, indent=2)

    difference_plot_path = _plot_results(
        fixed_results,
        two_group_result,
        method_specs,
    )
    power_plot_path = _plot_absolute_metric(
        fixed_results,
        two_group_result,
        method_specs,
        metric="Power_avg",
    )
    fwer_plot_path = _plot_absolute_metric(
        fixed_results,
        two_group_result,
        method_specs,
        metric="FWER",
    )
    print(f"Saved results: {json_path}")
    print(f"Saved difference plot: {difference_plot_path}")
    print(f"Saved power plot: {power_plot_path}")
    print(f"Saved FWER plot: {fwer_plot_path}")


if __name__ == "__main__":
    main()
