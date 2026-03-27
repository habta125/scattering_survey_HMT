from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .io import load_profile
from .preprocess import center_profile_on_peak, estimate_snr, subtract_baseline
from .fitter import fit_gaussian_only, fit_scattered_gaussian
from .qc import classify_fit


def resolve_value(row_value, loaded_value):
    """
    Prefer metadata.csv value if present, otherwise use loaded archive value.
    """
    if pd.notna(row_value):
        return float(row_value)
    if loaded_value is not None and np.isfinite(loaded_value):
        return float(loaded_value)
    return np.nan


def save_diagnostic_plot(
    profile: np.ndarray,
    fit_result: dict,
    out_png: Path,
    title: str,
) -> None:
    x = np.arange(len(profile))
    model = fit_result["model_full"]
    fit_idx = fit_result["fit_idx"]

    y_sub, baseline, rms = subtract_baseline(profile)
    model_sub = model - baseline

    peak = np.max(y_sub)
    if not np.isfinite(peak) or peak <= 0:
        peak = 1.0

    y_norm = y_sub / peak
    model_norm = model_sub / peak
    resid_norm = y_norm - model_norm
    rms_norm = rms / peak

    fig, axes = plt.subplots(
        3, 1, figsize=(10, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1, 1]}
    )

    axes[0].plot(x, y_norm, label="Observed profile")
    axes[0].plot(x, model_norm, label="Best-fit model")
    axes[0].axvspan(fit_idx[0], fit_idx[-1], alpha=0.15, label="Fit window")
    axes[0].set_ylabel("Normalized Intensity")
    axes[0].set_title(title)
    axes[0].legend()

    axes[1].plot(x, resid_norm)
    axes[1].axhline(0.0, linestyle="--")
    axes[1].set_ylabel("Residual")

    axes[2].plot(x, y_norm, label="Normalized signal")
    axes[2].axhline(3 * rms_norm, linestyle="--", label="3σ threshold")
    axes[2].set_xlabel("Phase (bin)")
    axes[2].set_ylabel("Normalized Intensity")
    axes[2].legend()

    plt.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def run_batch(
    data_dir: str | Path,
    metadata_df: pd.DataFrame,
    results_dir: str | Path,
    plots_dir: str | Path
) -> pd.DataFrame:
    data_dir = Path(data_dir)
    results_dir = Path(results_dir)
    plots_dir = Path(plots_dir)

    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for _, row in metadata_df.iterrows():
        name = row["name"]
        filename = row["filename"]
        filepath = data_dir / filename

        try:
            record = load_profile(
                filepath,
                dedisperse=True,
                remove_baseline=False,
                tscrunch=True,
                fscrunch=True,
                pscrunch=True,
            )

            profile_raw = record["profile"]
            nbin = record["nbin"]
            profile, phase_shift, peak_bin_before, target_bin = center_profile_on_peak(
                profile_raw, target_bin=nbin // 2
            )

            period_sec = resolve_value(row.get("period_sec", np.nan), record.get("period_sec", np.nan))
            freq_mhz = resolve_value(row.get("freq_mhz", np.nan), record.get("freq_mhz", np.nan))
            bandwidth_mhz = record.get("bandwidth_mhz", np.nan)
            mjd_start = record.get("mjd_start", np.nan)
            source_format = record.get("source_format", "unknown")

            if not np.isfinite(period_sec):
                raise ValueError(
                    f"No valid period_sec for {name}. "
                    "Provide it in metadata.csv or make sure the .ar file contains it."
                )

            bin_width_ms = period_sec * 1000.0 / nbin
            snr = estimate_snr(profile)

            fit_g = fit_gaussian_only(profile)
            fit_s = fit_scattered_gaussian(profile)

            delta_bic = fit_s["bic"] - fit_g["bic"]

            amp, mu, sigma_bin, tau_bin, baseline = fit_s["params"]
            amp_err, mu_err, sigma_bin_err, tau_bin_err, baseline_err = fit_s["errors"]

            tau_ms = tau_bin * bin_width_ms
            tau_ms_err = tau_bin_err * bin_width_ms
            sigma_ms = sigma_bin * bin_width_ms
            sigma_ms_err = sigma_bin_err * bin_width_ms

            flag = classify_fit(
                snr=snr,
                tau_bin=tau_bin,
                tau_bin_err=tau_bin_err,
                chi2_red=fit_s["chi2_red"],
                delta_bic=delta_bic,
            )

            title = (
                f"{name}: tau = {tau_bin:.2f} ± {tau_bin_err:.2f} bins"
                f" = {tau_ms:.2f} ± {tau_ms_err:.2f} ms, flag={flag}"
            )
            save_diagnostic_plot(profile, fit_s, plots_dir / f"{name}_fit.png", title)

            rows.append({
                "name": name,
                "filename": filename,
                "source_format": source_format,
                "period_sec": period_sec,
                "freq_mhz": freq_mhz,
                "bandwidth_mhz": bandwidth_mhz,
                "mjd_start": mjd_start,
                "nbin": nbin,
                "bin_width_ms": bin_width_ms,
                "snr": snr,
                "phase_shift": phase_shift,
                "peak_bin_before": peak_bin_before,
                "target_bin": target_bin,
                "mu_bin": mu,
                "mu_bin_err": mu_err,
                "sigma_bin": sigma_bin,
                "sigma_bin_err": sigma_bin_err,
                "sigma_ms": sigma_ms,
                "sigma_ms_err": sigma_ms_err,
                "tau_bin": tau_bin,
                "tau_bin_err": tau_bin_err,
                "tau_ms": tau_ms,
                "tau_ms_err": tau_ms_err,
                "baseline": baseline,
                "baseline_err": baseline_err,
                "chi2_red_scattered": fit_s["chi2_red"],
                "chi2_red_gaussian": fit_g["chi2_red"],
                "aic_scattered": fit_s["aic"],
                "aic_gaussian": fit_g["aic"],
                "bic_scattered": fit_s["bic"],
                "bic_gaussian": fit_g["bic"],
                "delta_bic": delta_bic,
                "flag": flag,
                "status": "OK",
            })

        except Exception as exc:
            rows.append({
                "name": name,
                "filename": filename,
                "source_format": "unknown",
                "period_sec": row.get("period_sec", np.nan),
                "freq_mhz": row.get("freq_mhz", np.nan),
                "phase_shift": np.nan,
                "peak_bin_before": np.nan,
                "target_bin": np.nan,
                "flag": "BAD_FIT",
                "status": f"ERROR: {exc}",
            })

    df = pd.DataFrame(rows)

    df.to_csv(results_dir / "fit_summary.csv", index=False)

    if "flag" in df.columns:
        df[df["flag"].isin(["STRONG_SCATTERING", "WEAK_SCATTERING"])].to_csv(
            results_dir / "good_fits.csv", index=False
        )
        df[df["flag"].isin(["BAD_FIT", "POOR_MODEL"])].to_csv(
            results_dir / "bad_fits.csv", index=False
        )
        df[df["flag"].isin(["UPPER_LIMIT", "MARGINAL", "LOW_SN"])].to_csv(
            results_dir / "upper_limits.csv", index=False
        )

    return df
