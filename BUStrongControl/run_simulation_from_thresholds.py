import importlib
import json
import os
from statistics import NormalDist

import numpy as np

cms_mod = importlib.import_module("cms_bottomup")
sim_mod = importlib.import_module("simulate_cms_normal")

importlib.reload(cms_mod)
importlib.reload(sim_mod)


# Edit these settings for each simulation run.
THRESHOLD_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cms_thresholds_K10.json")
std = NormalDist()
# Match the Pi_mix(0.7) data-generating alternative used by the K benchmark.
mu_alt = std.inv_cdf(0.05 / 10) - std.inv_cdf(0.7)
rho = 0.0
n_sim = 10000
seed = 123
n_jobs = -1


def main() -> None:
    with open(THRESHOLD_FILE, "r", encoding="utf-8") as f_in:
        threshold_payload = json.load(f_in)

    params = threshold_payload["params"]
    bu_specs = threshold_payload["bu_procedures"]

    K = int(params["K"])
    alpha = float(params["alpha"])
    alternative = params.get("alternative", "normal")
    beta_a = float(params.get("beta_a", 0.3))

    std = NormalDist()
    bu_procedures = []
    for spec in bu_specs:
        objective = spec["objective"]
        target_power = float(spec["target_power"])
        theta = std.inv_cdf(alpha / K) - std.inv_cdf(target_power)

        if alternative == "normal":
            def f(u, theta=theta):
                return sim_mod.g_theta_normal_p(u, theta)

            if objective == "bayes":
                def h(u, theta=theta):
                    return 1.0 + sim_mod.g_theta_normal_p(u, theta)
            else:
                def h(u):
                    return 1.0
        elif alternative == "beta_a1":
            def f(u):
                return sim_mod.beta_a1_density(u, beta_a)

            if objective == "bayes":
                def h(u):
                    return 1.0 + sim_mod.beta_a1_density(u, beta_a)
            else:
                def h(u):
                    return 1.0
        else:
            raise ValueError("alternative must be 'normal' or 'beta_a1'")

        bu_procedures.append(
            {
                "name": spec["name"],
                "objective": objective,
                "target_power": target_power,
                "theta": float(theta),
                "thresholds": np.array(spec["thresholds_t2_to_tK"], dtype=float),
                "f": f,
                "h": h,
            }
        )

    rng = np.random.default_rng(seed + 1000)
    configs = sim_mod.build_default_configs(K)

    results = []
    for i, true_null in enumerate(configs, start=1):
        n_false = int((~true_null).sum())
        print(f"Running config {i}/{len(configs)} with n_false={n_false}, rho={rho}")

        if alternative == "normal":
            mu = np.where(true_null, 0.0, mu_alt)
            stats = sim_mod.simulate_config_multi_bu(
                n_sim=n_sim,
                mu=mu,
                true_null=true_null,
                rho=rho,
                bu_procedures=bu_procedures,
                alpha=alpha,
                rng=rng,
                n_jobs=n_jobs,
            )
        else:
            stats = sim_mod.simulate_config_beta_a1_multi_bu(
                n_sim=n_sim,
                K=K,
                true_null=true_null,
                rho=rho,
                a=beta_a,
                bu_procedures=bu_procedures,
                alpha=alpha,
                rng=rng,
                n_jobs=n_jobs,
            )

        row = {
            "K": K,
            "n_false": n_false,
            "rho": rho,
            **stats,
        }
        results.append(row)
        print(row)

    payload = {
        "threshold_file": THRESHOLD_FILE,
        "threshold_params": params,
        "simulation_params": {
            "mu_alt": mu_alt,
            "rho": rho,
            "n_sim": n_sim,
            "seed": seed,
            "n_jobs": n_jobs,
            "alternative": alternative,
            "beta_a": beta_a,
            "timing_clock": "time.process_time",
            "timing_definition": "mean CPU execution time per procedure call",
        },
        "bu_procedures": bu_specs,
        "results": results,
    }

    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_dir, f"cms_compare_rho_{str(rho).replace('.', 'p')}.json")
    with open(json_path, "w", encoding="utf-8") as f_out:
        json.dump(payload, f_out, indent=2)

    plot_prefix = os.path.join(base_dir, f"cms_compare_rho_{str(rho).replace('.', 'p')}")
    plot_paths = sim_mod.plot_simulation_results(results, plot_prefix)

    print(f"Saved simulation results to: {json_path}")
    print(plot_paths)


if __name__ == "__main__":
    main()
