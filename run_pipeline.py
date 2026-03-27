from pathlib import Path

from src.io import load_metadata
from src.batch import run_batch


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    results_dir = base_dir / "results"
    plots_dir = base_dir / "plots"
    metadata_csv = base_dir / "metadata.csv"

    metadata_df = load_metadata(metadata_csv)
    summary = run_batch(data_dir, metadata_df, results_dir, plots_dir)

    print("\nDone.\n")

    cols = [
        "name",
        "filename",
        "source_format",
        "flag",
        "tau_bin",
        "tau_ms",
        "chi2_red_scattered",
        "delta_bic",
        "status",
    ]
    existing_cols = [c for c in cols if c in summary.columns]
    print(summary[existing_cols].to_string(index=False))


if __name__ == "__main__":
    main()
