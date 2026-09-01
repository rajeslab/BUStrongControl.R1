import importlib
import json
import os
from statistics import NormalDist

cms_mod = importlib.import_module("cms_bottomup")
sim_mod = importlib.import_module("simulate_cms_normal")

importlib.reload(cms_mod)
importlib.reload(sim_mod)


# Edit these settings once, then run this file to save thresholds.
K = 5
alpha = 0.05
target_powers = [0.3, 0.7, 0.95]
beta_a = 0.3
B_threshold = 500000
seed = 123
n_jobs = -1
alternative = "normal"   # or "beta_a1"
chunk_size = 2000
null_correlation_phi = 0.8   # use phi > 0 for equicorrelated Gaussian-null calibration
procedure_target_powers = {
    "bayes": [0.3, 0.7, 0.95],
    "pi1": [0.3, 0.7, 0.95],
}


def main() -> None:
    std = NormalDist()
    bu_procedures = []
    for procedure_type, target_powers in procedure_target_powers.items():
        for target_power in target_powers:
            theta = std.inv_cdf(alpha / K) - std.inv_cdf(target_power)
            if alternative == "normal":
                def f(u, theta=theta):
                    return sim_mod.g_theta_normal_p(u, theta)

                if procedure_type == "bayes":
                    def h(u, theta=theta):
                        return 1.0 + sim_mod.g_theta_normal_p(u, theta)
                elif procedure_type == "pi1":
                    def h(u):
                        return 1.0
                else:
                    raise ValueError("procedure type must be 'bayes' or 'pi1'")
            elif alternative == "beta_a1":
                def f(u):
                    return sim_mod.beta_a1_density(u, beta_a)

                if procedure_type == "bayes":
                    def h(u):
                        return 1.0 + sim_mod.beta_a1_density(u, beta_a)
                elif procedure_type == "pi1":
                    def h(u):
                        return 1.0
                else:
                    raise ValueError("procedure type must be 'bayes' or 'pi1'")
            else:
                raise ValueError("alternative must be 'normal' or 'beta_a1'")

            print(f"Estimating thresholds for BU_{procedure_type} target_power={target_power}")
            thresholds = cms_mod.estimate_cms_thresholds(
                K=K,
                B=B_threshold,
                f=f,
                h=h,
                alpha=alpha,
                seed=seed,
                chunk_size=chunk_size,
                n_jobs=n_jobs,
                null_correlation_phi=null_correlation_phi,
                verbose=True,
            )
            bu_procedures.append(
                {
                    "name": f"BU_{procedure_type}_tp_{str(target_power).replace('.', 'p')}",
                    "objective": procedure_type,
                    "target_power": target_power,
                    "theta": float(theta),
                    "thresholds_t2_to_tK": thresholds.tolist(),
                }
            )

    payload = {
        "params": {
            "K": K,
            "alpha": alpha,
            "procedure_target_powers": procedure_target_powers,
            "beta_a": beta_a,
            "alternative": alternative,
            "null_correlation_phi": null_correlation_phi,
            "B_threshold": B_threshold,
            "seed": seed,
            "n_jobs": n_jobs,
            "chunk_size": chunk_size,
        },
        "bu_procedures": bu_procedures,
    }

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cms_thresholds.json")
    with open(out_path, "w", encoding="utf-8") as f_out:
        json.dump(payload, f_out, indent=2)

    for procedure in bu_procedures:
        print(procedure["name"], procedure["thresholds_t2_to_tK"])
    print(f"Saved thresholds to: {out_path}")


if __name__ == "__main__":
    main()
