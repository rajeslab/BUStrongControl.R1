import importlib
import json
import os


sim_mod = importlib.import_module("simulate_cms_normal")
importlib.reload(sim_mod)


# Edit these two paths as needed.
JSON_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cms_compare_rho_0p3.json")
PLOT_PREFIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cms_compare_rho_0p3")


def main() -> None:
    with open(JSON_FILE, "r", encoding="utf-8") as f_in:
        payload = json.load(f_in)

    results = payload["results"]
    plot_paths = sim_mod.plot_simulation_results(results, PLOT_PREFIX)

    for path in plot_paths:
        print(f"Saved plot to: {path}")


if __name__ == "__main__":
    main()
