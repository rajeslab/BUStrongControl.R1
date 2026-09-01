import importlib
import json
import os


ih_mod = importlib.import_module("improved_hommel")
importlib.reload(ih_mod)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_THRESHOLD_FILE = os.path.join(BASE_DIR, "cms_thresholds_K5.json")
OUTPUT_FILE = os.path.join(BASE_DIR, "improved_hommel_thresholds_K5.json")

B_OVERRIDE = None  # None uses B_threshold from the source threshold JSON.
CHUNK_SIZE = 20_000
SEED_OVERRIDE = None  # None uses the source threshold seed.
QUANTILE_METHOD = "higher"
MERGE_INTO_SOURCE = True


def main() -> None:
    with open(SOURCE_THRESHOLD_FILE, "r", encoding="utf-8") as f_in:
        source = json.load(f_in)

    params = source["params"]
    K = int(params["K"])
    alpha = float(params["alpha"])
    B = int(params["B_threshold"] if B_OVERRIDE is None else B_OVERRIDE)
    seed = int(params["seed"] if SEED_OVERRIDE is None else SEED_OVERRIDE)
    null_correlation = float(params.get("null_correlation_phi", 0.0))
    target_powers = sorted(
        {
            float(value)
            for values in params["procedure_target_powers"].values()
            for value in values
        }
    )

    procedures = ih_mod.estimate_improved_hommel_thresholds(
        K=K,
        B=B,
        alpha=alpha,
        target_powers=target_powers,
        seed=seed,
        chunk_size=CHUNK_SIZE,
        null_correlation=null_correlation,
        quantile_method=QUANTILE_METHOD,
        verbose=True,
    )

    output = {
        "params": {
            "K": K,
            "alpha": alpha,
            "target_powers": target_powers,
            "B_threshold": B,
            "seed": seed,
            "chunk_size": CHUNK_SIZE,
            "null_correlation": null_correlation,
            "quantile_method": QUANTILE_METHOD,
            "candidate_method": "leave-one-out Hommel",
            "gate": "score > threshold",
            "source_threshold_file": SOURCE_THRESHOLD_FILE,
        },
        "procedures": procedures,
    }
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        json.dump(output, f_out, indent=2)

    print(f"Saved improved-Hommel thresholds to: {OUTPUT_FILE}")

    if MERGE_INTO_SOURCE:
        source["improved_hommel"] = output
        with open(SOURCE_THRESHOLD_FILE, "w", encoding="utf-8") as f_out:
            json.dump(source, f_out, indent=2)
        print(
            "Updated improved_hommel section in: "
            f"{SOURCE_THRESHOLD_FILE}"
        )


if __name__ == "__main__":
    main()
