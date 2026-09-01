"""Run the all-configuration BU-versus-Hommel comparison.

Edit the values in the PARAMETERS section, then run:

    python3 run_configuration_table.py
"""

from pathlib import Path

import numpy as np

from bu_procedure import fit_bu_procedure
from configuration_table import (
    build_table_rows,
    formatted_rows,
    render_console_table,
    render_latex_table,
    write_csv,
)


# ---------------------------------------------------------------------------
# PARAMETERS: edit these values for your experiment
# ---------------------------------------------------------------------------

SIGMA = np.array(
    [
        [2.5721, 0.3355, 0.3355],
        [0.3355, 2.5721, 0.3355],
        [0.3355, 0.3355, 0.7828],
    ],
    dtype=float,
)

THETA_ALT = 2.0
ALPHA = 0.05

# Number of null simulations used to calibrate each BU threshold.
B = 200_000

# Number of simulations for each configuration h in the output table.
N_SIM = 100_000

N_JOBS = 8
CALIBRATION_SEED = 123
TABLE_SEED = 456
DECIMAL_DIGITS = 4

# Match the example: show standard errors only for global-null FWER.
# Other choices are "none" and "all".
STANDARD_ERRORS = "global-null"

OUTPUT_CSV = Path("configuration_table.csv")
OUTPUT_TEX = Path("configuration_table.tex")


def main() -> None:
    print("Calibrating BU thresholds...")
    fit = fit_bu_procedure(
        Sigma=SIGMA,
        theta_alt=THETA_ALT,
        alpha=ALPHA,
        B=B,
        random_seed=CALIBRATION_SEED,
        n_jobs=N_JOBS,
    )

    print("Simulating every null/alternative configuration...")
    rows = build_table_rows(
        fit_object=fit,
        n_sim=N_SIM,
        random_seed=TABLE_SEED,
        n_jobs=N_JOBS,
    )
    table = formatted_rows(
        rows,
        digits=DECIMAL_DIGITS,
        standard_errors=STANDARD_ERRORS,
    )

    write_csv(rows, OUTPUT_CSV)
    OUTPUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_TEX.write_text(render_latex_table(table))

    print()
    print(render_console_table(table))
    print(f"\nRaw results: {OUTPUT_CSV.resolve()}")
    print(f"LaTeX table: {OUTPUT_TEX.resolve()}")


if __name__ == "__main__":
    main()
