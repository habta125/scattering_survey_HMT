import matplotlib.pyplot as plt
import numpy as np

from .utils import estimate_baseline_and_rms


def save_quicklook_plot(clean_cube, profile, dynspec, start, end, outpath, title=""):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax in axes.ravel():
        ax.grid(False)

    axes[0, 0].imshow(clean_cube.mean(0), aspect="auto", origin="lower", interpolation="nearest")
    axes[0, 0].set_xlabel("Pulse Phase (bins)")
    axes[0, 0].set_ylabel("Frequency (Channels)") # Need to change with your observation frequency (MHz)
    axes[0, 0].set_title("Frequency Domain: mean over time")

    axes[0, 1].imshow(clean_cube.mean(1), aspect="auto", origin="lower", interpolation="nearest")
    axes[0, 1].set_xlabel("Pulse Phase (bins)")
    axes[0, 1].set_ylabel("Subintegration")
    axes[0, 1].set_title("Time Domain: mean over frequency")

    axes[1, 0].plot(profile / max(profile), lw=1.2)
    axes[1, 0].axvspan(start, end, color="tab:orange", alpha=0.2)
    axes[1, 0].set_xlabel("Pulse Phase (bins)")
    axes[1, 0].set_ylabel("Normalized Intensity")
    axes[1, 0].set_title("Integrated pulse profile")

    valid = dynspec[np.isfinite(dynspec)]
    if valid.size > 0:
        mn = np.median(valid)
        mad = np.median(np.abs(valid - mn))
        std = 1.4826 * mad if mad > 0 else np.std(valid)
        if not np.isfinite(std) or std <= 0:
            std = 1.0
        vmin = mn - 3 * std
        vmax = mn + 5 * std
    else:
        vmin, vmax = None, None

    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="black")
    axes[1, 1].imshow(
        dynspec.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )
    axes[1, 1].set_xlabel("Time (bins)")
    axes[1, 1].set_ylabel("Frequency (MHz)")
    axes[1, 1].set_title("Dynamic spectrum")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_fit_plot(profile, fit_result, outpath, title=""):
    y = np.asarray(profile, dtype=float)
    x = np.arange(len(y))
    fit_idx = fit_result["fit_idx"]

    # --- baseline ---
    baseline, rms = estimate_baseline_and_rms(y)
    y_bs = y - baseline
    model_bs = np.asarray(fit_result["model_full"], dtype=float) - baseline

    # --- normalization ---
    peak = np.max(y_bs)
    if not np.isfinite(peak) or peak <= 0:
        peak = 1.0

    y_norm = y_bs / peak
    model_norm = model_bs / peak
    resid_norm = y_norm - model_norm
    rms_norm = rms / peak

    fig, axes = plt.subplots(
        3, 1, figsize=(9, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1, 1]}
    )

    # --- Top: profile ---
    axes[0].plot(x, y_norm, label="Profile")
    axes[0].plot(x, model_norm, label=f"{fit_result['model_name']}")
    axes[0].axvspan(fit_idx[0], fit_idx[-1], alpha=0.15, label="fit window")
    axes[0].legend()
    axes[0].set_ylabel("Normalized Intensity")
    axes[0].set_title(
        f"{title} | {fit_result['selected_model']} | "
        f"tau = {fit_result['tau_bin']:.2f} ± {fit_result['tau_bin_err']:.2f} bins = "
        f"{fit_result['tau_ms']:.2f} ± {fit_result['tau_ms_err']:.2f} ms | "
        f"{fit_result['classification']}"
    )

    # --- Middle: residual ---
    axes[1].plot(x, resid_norm)
    axes[1].axhline(0, ls="--")
    axes[1].set_ylabel("Residual")

    # --- Bottom: normalized signal ---
    axes[2].plot(x, y_norm, label="Signal")
    axes[2].axhline(3 * rms_norm, ls="--", label="3σ")
    axes[2].legend()
    axes[2].set_xlabel("Pulse Phase (bins)")
    axes[2].set_ylabel("Normalized Intensity")

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)

def save_trace_plot(fit_result, outpath, title=""):
    chain = fit_result["mcmc_chain"]
    labels = fit_result["param_labels"]
    nsteps, nwalkers, ndim = chain.shape

    fig, axes = plt.subplots(ndim, 1, figsize=(10, 1.8 * ndim), sharex=True)
    if ndim == 1:
        axes = [axes]
    x = np.arange(nsteps)
    for i in range(ndim):
        for j in range(nwalkers):
            axes[i].plot(x, chain[:, j, i], alpha=0.25, lw=0.5)
        axes[i].set_ylabel(labels[i])
    axes[-1].set_xlabel("Step")
    fig.suptitle(title if title else "MCMC traces")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
