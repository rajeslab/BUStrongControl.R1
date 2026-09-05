# Paper Reproducibility Code

This repository contains the Python code used to reproduce the simulation figures and supplementary results for the paper.

## Setup

Run all examples from the repository root:

```bash
cd /path/to/BUStrongControl
```

Install the required Python packages:

```bash
python3 -m pip install numpy matplotlib joblib
```

The examples below use the project directory as the working directory:

```python
from pathlib import Path

project_dir = Path.cwd()
```

## 1. Figures 3 and S2

First generate `cms_compare_rho_0p0.json` in each of the following study directories (using the method of section 3.1 by changing the relevant parameters and file names):

```text
K10dg03/
K10dg07/
K5dg03/
K5dg07/
```

Then run:

```python
import importlib

tg_mod = importlib.import_module("add_twogroup_dgm_rho0")
importlib.reload(tg_mod)

tg_mod.N_SIM_OVERRIDE = 200_000
tg_mod.CHUNK_SIZE = 2_000
tg_mod.SEED_OFFSET = 2_000
tg_mod.N_JOBS_OVERRIDE = -1
tg_mod.FORCE_RECOMPUTE = True

tg_mod.main()
```

This generates:

```text
K10dg03/cms_compare_rho_0p0_twogroup_dgm_fwer.png
K10dg03/cms_compare_rho_0p0_twogroup_dgm_power.png

K5dg03/cms_compare_rho_0p0_twogroup_dgm_fwer.png
K5dg03/cms_compare_rho_0p0_twogroup_dgm_power.png

K10dg07/cms_compare_rho_0p0_twogroup_dgm_fwer.png
K10dg07/cms_compare_rho_0p0_twogroup_dgm_power.png

K5dg07/cms_compare_rho_0p0_twogroup_dgm_fwer.png
K5dg07/cms_compare_rho_0p0_twogroup_dgm_power.png
```

Set `FORCE_RECOMPUTE = False` to redraw the figures from existing `cms_compare_rho_0p0_twogroup_dgm.json` files without rerunning the two-group simulations.

## 2. Computational Complexity

### 2.1 Figure S7

Data file used (generate it using instructions in section 3.1): 
```
cms_thresholds_K10.json
```

```python
import importlib

fixed_mod = importlib.import_module("run_simulation_from_thresholds")
importlib.reload(fixed_mod)

fixed_mod.rho = 0.0
fixed_mod.n_sim = 10_000
fixed_mod.seed = 123
fixed_mod.n_jobs = -1

fixed_mod.main()
```

This generates:

```text
cms_compare_rho_0p0.json
cms_compare_rho_0p0_fwer.png
cms_compare_rho_0p0_power.png
cms_compare_rho_0p0_time.png
cms_compare_rho_0p0_combined.png
```

### 2.2 Figure S8

```python
import importlib

runtime_mod = importlib.import_module("benchmark_runtime_by_dimension")
importlib.reload(runtime_mod)

runtime_mod.K_MIN = 2
runtime_mod.K_MAX = 10
runtime_mod.N_SIM = 10_000
runtime_mod.RHO = 0.0
runtime_mod.SEED = 123
runtime_mod.N_JOBS = -1
runtime_mod.DATA_TARGET_POWER = 0.7
runtime_mod.ALTERNATIVE_PROBABILITY = 0.5

runtime_mod.main()
```

This generates:

```text
runtime_by_dimension_pimix_0p7.png
runtime_by_dimension_pimix_0p7_fwer.png
runtime_by_dimension_pimix_0p7_power_tpr.png
runtime_by_dimension_pimix_0p7_combined.png
runtime_by_dimension_pimix_0p7.json
```

## 3. Dependence

### 3.1 Independent BU Under Dependence: Figures S4, S5, and S6

#### Generate the independent-null thresholds

```python
import importlib
import json
import shutil
from pathlib import Path

project_dir = Path.cwd()

threshold_mod = importlib.import_module("compute_thresholds_once")
importlib.reload(threshold_mod)

threshold_mod.K = 10
threshold_mod.alpha = 0.05
threshold_mod.procedure_target_powers = {
    "bayes": [0.3, 0.7, 0.95],
    "pi1": [0.3, 0.7, 0.95],
}
threshold_mod.alternative = "normal"
threshold_mod.B_threshold = 1_000_000
threshold_mod.seed = 123
threshold_mod.n_jobs = -1
threshold_mod.chunk_size = 2_000
threshold_mod.null_correlation_phi = 0.0

# Generates cms_thresholds.json.
threshold_mod.main()

source_file = project_dir / "cms_thresholds.json"
destination_file = project_dir / "cms_thresholds_K10.json"

with source_file.open("r", encoding="utf-8") as f:
    payload = json.load(f)

if payload["params"]["K"] != 10:
    raise ValueError("The generated threshold file is not for K=10.")

shutil.copy2(source_file, destination_file)
print("Created:", destination_file)
```

This creates:

```text
cms_thresholds_K10.json
```

#### Run the simulation

```python
import importlib
from pathlib import Path
from statistics import NormalDist

project_dir = Path.cwd()

run_mod = importlib.import_module("run_simulation_from_thresholds")
importlib.reload(run_mod)

K = 10
alpha = 0.05

# Change to 0.7 or 0.95 for the other data-generating alternatives.
data_target_power = 0.3

std = NormalDist()

run_mod.THRESHOLD_FILE = str(project_dir / "cms_thresholds_K10.json")
run_mod.mu_alt = (
    std.inv_cdf(alpha / K)
    - std.inv_cdf(data_target_power)
)

# Repeat with rho equal to -0.1, 0.0, 0.3, and 0.7.
run_mod.rho = -0.1
run_mod.n_sim = 200_000
run_mod.seed = 123
run_mod.n_jobs = -1

run_mod.main()
```

For `rho = -0.1`, this creates:

```text
cms_compare_rho_-0p1.json
cms_compare_rho_-0p1_fwer.png
cms_compare_rho_-0p1_power.png
cms_compare_rho_-0p1_time.png
cms_compare_rho_-0p1_combined.png
```

Copy the generated JSON file into the matching study directory before generating study-specific plots. For example, the target-power `0.3`, `K=10` result belongs in `K10dg03/`.

#### Regenerate plots from a saved JSON file

```python
import importlib
import json
from pathlib import Path

project_dir = Path.cwd()

sim_mod = importlib.import_module("simulate_cms_normal")
importlib.reload(sim_mod)

json_file = project_dir / "K10dg03" / "cms_compare_rho_-0p1.json"
plot_prefix = project_dir / "K10dg03" / "cms_compare_rho_-0p1_custom"

with json_file.open("r", encoding="utf-8") as f:
    payload = json.load(f)

plot_paths = sim_mod.plot_simulation_results(
    payload["results"],
    str(plot_prefix),
)

for path in plot_paths:
    print("Saved:", path)
```

The current plotting function creates:

```text
K10dg03/cms_compare_rho_-0p1_custom_fwer.png
K10dg03/cms_compare_rho_-0p1_custom_power.png
K10dg03/cms_compare_rho_-0p1_custom_time.png
K10dg03/cms_compare_rho_-0p1_custom_combined.png
```

### 3.2 Adapting to Dependence: Table S1

The reported settings are:

```text
Threshold-calibration samples: B = 200,000
Simulation samples: N_SIM = 100,000
```

Run:

```bash
python3 run_configuration_table.py
```

It generates:

```bash
configuration_table.csv
configuration_table.tex
```

## 4. Figure S3

### 4.1 Generate the K=10 Improved-Hommel Thresholds

```python
import importlib
from pathlib import Path

project_dir = Path.cwd()

threshold_mod = importlib.import_module(
    "compute_improved_hommel_thresholds"
)
importlib.reload(threshold_mod)

threshold_mod.SOURCE_THRESHOLD_FILE = str(
    project_dir / "cms_thresholds_K10.json"
)
threshold_mod.OUTPUT_FILE = str(
    project_dir / "improved_hommel_thresholds_K10.json"
)

threshold_mod.B_OVERRIDE = 1_000_000
threshold_mod.SEED_OVERRIDE = 123
threshold_mod.CHUNK_SIZE = 20_000
threshold_mod.QUANTILE_METHOD = "higher"

# Do not modify cms_thresholds_K10.json.
threshold_mod.MERGE_INTO_SOURCE = False

threshold_mod.main()
```

This creates:

```text
improved_hommel_thresholds_K10.json
```

### 4.2 Generate Figure S3

```python
import importlib
from pathlib import Path

project_dir = Path.cwd()

plot_mod = importlib.import_module("plot_improved_hommel_power_gain")
importlib.reload(plot_mod)

plot_mod.K = 10
plot_mod.ALPHA = 0.05
plot_mod.DATA_TARGET_POWER = 0.3
plot_mod.ALTERNATIVE_PROBABILITY = 0.5
plot_mod.N_SIM = 200_000
plot_mod.CHUNK_SIZE = 20_000
plot_mod.SEED = 123

plot_mod.THRESHOLD_FILE = str(
    project_dir / "improved_hommel_thresholds_K10.json"
)
plot_mod.OUTPUT_PREFIX = str(
    project_dir / "improved_hommel_power_gain_K10_tp_0p3"
)

plot_mod.main()
```

This generates:

```text
improved_hommel_power_gain_K10_tp_0p3.json
improved_hommel_power_gain_K10_tp_0p3.png
improved_hommel_power_gain_K10_tp_0p3_power.png
improved_hommel_power_gain_K10_tp_0p3_fwer.png
```

## 5. Generate Table 1 and Table 2

P-values of 248 outcomes (248 by 5 matrix):

```text
pvmat.csv
pvmat.xlsx
```

Use the following code to merge "improved_hommel_thresholds_K5.json" into "cms_thresholds_K5.json":

```bash
import json
from pathlib import Path

PROJECT_DIR = Path.cwd()

cms_file = PROJECT_DIR / "cms_thresholds_K5.json"
improved_file = PROJECT_DIR / "improved_hommel_thresholds_K5.json"

with cms_file.open("r", encoding="utf-8") as f:
    cms_payload = json.load(f)

with improved_file.open("r", encoding="utf-8") as f:
    improved_payload = json.load(f)

if int(cms_payload["params"]["K"]) != int(
    improved_payload["params"]["K"]
):
    raise ValueError("K differs between the threshold files.")

if float(cms_payload["params"]["alpha"]) != float(
    improved_payload["params"]["alpha"]
):
    raise ValueError("Alpha differs between the threshold files.")

if "procedures" not in improved_payload:
    raise ValueError(
        "The improved-Hommel file has no procedures section."
    )

cms_payload["improved_hommel"] = improved_payload

with cms_file.open("w", encoding="utf-8") as f:
    json.dump(cms_payload, f, indent=2)

print("Updated:", cms_file)
print("Top-level keys:", list(cms_payload))
print(
    "Improved-Hommel procedures:",
    len(cms_payload["improved_hommel"]["procedures"]),
)
```

Run the notebook:

```bash
cochrane_example.ipynb
```


This generates:

```bash
cochrane_table1_bu_mix_vs_gou.csv
cochrane_table2_discovery_summary.csv
```

## 6. Rejection Regions: Figure 2 and Figure S9

Run the notebook:

```bash
Final_Codes_grid.ipynb
```

Files needed to run this notebook (if not present, the notebook will create it):

```bash
cms_thresholds_K3.json
```

Output files:

```
rejection_policy_third_coordinate.png
rejection_count_regions_gou_vs_bu.png
```

## 7. Recovery Dataset

Install ```multcomp``` package in R.

Run the R-script:

```bash
recovery_dataset.R
```

Output: p-values and covariance matrix.
