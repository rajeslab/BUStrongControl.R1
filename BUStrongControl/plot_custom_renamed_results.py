import importlib
import json
import copy

sim_mod = importlib.import_module("simulate_cms_normal")
importlib.reload(sim_mod)

json_file = "/Users/rajeshkarmakar/Downloads/BUStrongControl/K10dg03/cms_compare_rho_-0p1.json"
plot_prefix = "/Users/rajeshkarmakar/Downloads/BUStrongControl/K10dg03/cms_compare_rho_-0p1_custom"

with open(json_file, "r", encoding="utf-8") as f:
    payload = json.load(f)

results = copy.deepcopy(payload["results"])

rename_map = {
    "BU_bayes_tp_0p3": r"BU $\Pi_{mix}(0.3)$",
    "BU_bayes_tp_0p7": r"BU $\Pi_{mix}(0.7)$",
    "BU_bayes_tp_0p95": r"BU $\Pi_{mix}(0.95)$",
    "BU_pi1_tp_0p3": r"BU $\Pi_{1}(0.3)$",
    "BU_pi1_tp_0p7": r"BU $\Pi_{1}(0.7)$",
    "BU_pi1_tp_0p95": r"BU $\Pi_{1}(0.95)$",
    "GTXR": "Gou",
    "Holm": "Holm",
    "Hommel": "Hommel",
}

for row in results:
    updates = {}
    to_delete = []

    for key, value in row.items():
        if key.endswith("_FWER"):
            base = key[:-5]
            if base in rename_map:
                updates[f"{rename_map[base]}_FWER"] = value
                to_delete.append(key)
        elif key.endswith("_Power_any"):
            base = key[:-10]
            if base in rename_map:
                updates[f"{rename_map[base]}_Power_any"] = value
                to_delete.append(key)
        elif key.endswith("_Power_avg"):
            base = key[:-10]
            if base in rename_map:
                updates[f"{rename_map[base]}_Power_avg"] = value
                to_delete.append(key)

    for key in to_delete:
        del row[key]
    row.update(updates)

plot_paths = sim_mod.plot_simulation_results(results, plot_prefix)
print(plot_paths)
