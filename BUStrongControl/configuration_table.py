"""Generate the BU-versus-Hommel FWER/TPR table for all configurations."""

from __future__ import annotations

import argparse
import csv
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from bu_procedure import (
    BUProcedureFit,
    fit_bu_procedure,
    simulate_all_configurations_competition,
)


DEFAULT_SIGMA = np.array(
    [
        [2.5721, 0.3355, 0.3355],
        [0.3355, 2.5721, 0.3355],
        [0.3355, 0.3355, 0.7828],
    ],
    dtype=float,
)


def ordered_configurations(K: int) -> List[Tuple[int, ...]]:
    """Order configurations by number and then positions of non-nulls."""
    configurations: List[Tuple[int, ...]] = []
    for number_nonnull in range(K + 1):
        for nonnull_indices in combinations(range(K), number_nonnull):
            h = [0] * K
            for index in nonnull_indices:
                h[index] = 1
            configurations.append(tuple(h))
    return configurations


def build_table_rows(
    fit_object: BUProcedureFit,
    n_sim: int,
    random_seed: int | None = None,
    n_jobs: int | None = None,
) -> List[Dict[str, object]]:
    """Simulate both procedures and return one summary row for every h."""
    results = simulate_all_configurations_competition(
        fit_object=fit_object,
        n_sim=n_sim,
        random_seed=random_seed,
        n_jobs=n_jobs,
    )
    K = fit_object.Sigma.shape[0]
    rows: List[Dict[str, object]] = []

    for h in ordered_configurations(K):
        bu = results[h]["bu"]
        hommel = results[h]["hommel"]
        number_nonnull = sum(h)
        has_null = number_nonnull < K
        has_alternative = number_nonnull > 0
        rows.append(
            {
                "h": h,
                "fwer_bu": float(bu["fwer"]) if has_null else None,
                "fwer_bu_se": float(bu["fwer_se"]) if has_null else None,
                "fwer_hommel": float(hommel["fwer"]) if has_null else None,
                "fwer_hommel_se": (
                    float(hommel["fwer_se"]) if has_null else None
                ),
                "tpr_bu": float(bu["average_power"]) if has_alternative else None,
                "tpr_bu_se": (
                    float(bu["average_power_se"]) if has_alternative else None
                ),
                "tpr_hommel": (
                    float(hommel["average_power"]) if has_alternative else None
                ),
                "tpr_hommel_se": (
                    float(hommel["average_power_se"])
                    if has_alternative
                    else None
                ),
            }
        )
    return rows


def _format_h(h: Sequence[int]) -> str:
    return "(" + ",".join(str(bit) for bit in h) + ")"


def _format_estimate(
    value: object,
    standard_error: object,
    digits: int,
    include_standard_error: bool,
) -> str:
    if value is None:
        return ""
    estimate = f"{float(value):.{digits}f}"
    if include_standard_error and standard_error is not None:
        estimate += f"({float(standard_error):.{digits}f})"
    return estimate


def formatted_rows(
    rows: Sequence[Dict[str, object]],
    digits: int = 4,
    standard_errors: str = "global-null",
) -> List[List[str]]:
    """Format rows for display, following the example table by default."""
    if standard_errors not in {"none", "global-null", "all"}:
        raise ValueError("standard_errors must be 'none', 'global-null', or 'all'.")

    output: List[List[str]] = []
    for row in rows:
        h = tuple(row["h"])
        global_null = sum(h) == 0
        show_fwer_se = standard_errors == "all" or (
            standard_errors == "global-null" and global_null
        )
        show_tpr_se = standard_errors == "all"
        output.append(
            [
                _format_h(h),
                _format_estimate(
                    row["fwer_bu"], row["fwer_bu_se"], digits, show_fwer_se
                ),
                _format_estimate(
                    row["fwer_hommel"],
                    row["fwer_hommel_se"],
                    digits,
                    show_fwer_se,
                ),
                _format_estimate(
                    row["tpr_bu"], row["tpr_bu_se"], digits, show_tpr_se
                ),
                _format_estimate(
                    row["tpr_hommel"],
                    row["tpr_hommel_se"],
                    digits,
                    show_tpr_se,
                ),
            ]
        )
    return output


def render_console_table(formatted: Sequence[Sequence[str]]) -> str:
    headers = ["h", "FWER BU", "FWER Hommel", "TPR BU", "TPR Hommel"]
    widths = [
        max(len(headers[column]), *(len(row[column]) for row in formatted))
        for column in range(len(headers))
    ]

    def render_row(row: Sequence[str]) -> str:
        return "  ".join(
            value.ljust(widths[0]) if column == 0 else value.rjust(widths[column])
            for column, value in enumerate(row)
        )

    separator = "  ".join("-" * width for width in widths)
    return "\n".join([render_row(headers), separator, *(render_row(row) for row in formatted)])


def render_latex_table(formatted: Sequence[Sequence[str]]) -> str:
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"& \multicolumn{2}{c}{FWER} & \multicolumn{2}{c}{TPR} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"$\boldsymbol{h}$ & BU & Hommel & BU & Hommel \\",
        r"\midrule",
    ]
    for h, fwer_bu, fwer_hommel, tpr_bu, tpr_hommel in formatted:
        h_latex = "$" + h.replace(",", ",\\,") + "$"
        lines.append(
            f"{h_latex} & {fwer_bu} & {fwer_hommel} & {tpr_bu} & {tpr_hommel} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def write_csv(rows: Sequence[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["h"] = _format_h(row["h"])
            writer.writerow(csv_row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an all-configuration FWER/TPR table for BU and Hommel."
    )
    parser.add_argument("--theta-alt", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--B", type=int, default=10_000)
    parser.add_argument("--n-sim", type=int, default=100_000)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--random-seed", type=int, default=123)
    parser.add_argument("--digits", type=int, default=4)
    parser.add_argument(
        "--standard-errors",
        choices=["none", "global-null", "all"],
        default="global-null",
    )
    parser.add_argument(
        "--output-csv", type=Path, default=Path("configuration_table.csv")
    )
    parser.add_argument(
        "--output-tex", type=Path, default=Path("configuration_table.tex")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fit = fit_bu_procedure(
        Sigma=DEFAULT_SIGMA,
        theta_alt=args.theta_alt,
        alpha=args.alpha,
        B=args.B,
        random_seed=args.random_seed,
        n_jobs=args.n_jobs,
    )
    rows = build_table_rows(
        fit_object=fit,
        n_sim=args.n_sim,
        random_seed=args.random_seed + 1,
        n_jobs=args.n_jobs,
    )
    table = formatted_rows(
        rows,
        digits=args.digits,
        standard_errors=args.standard_errors,
    )
    write_csv(rows, args.output_csv)
    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    args.output_tex.write_text(render_latex_table(table))

    print(render_console_table(table))
    print(f"\nWrote raw results to {args.output_csv}")
    print(f"Wrote LaTeX table to {args.output_tex}")


if __name__ == "__main__":
    main()
