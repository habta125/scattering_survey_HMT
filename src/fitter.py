import numpy as np
from scipy.optimize import curve_fit

from .models import gaussian_only_model_full, scattered_gaussian_model_full
from .preprocess import estimate_baseline_and_rms, find_fit_window
from .utils import reduced_chi2, aic, bic


def fit_gaussian_only(profile: np.ndarray) -> dict:
    y = np.asarray(profile, dtype=float)
    nbin = len(y)

    baseline0, rms0 = estimate_baseline_and_rms(y)
    y0 = y - baseline0

    fit_idx = find_fit_window(y, nsigma=3.0, pad=15)
    x_fit = fit_idx.astype(float)
    y_fit = y[fit_idx]

    amp0 = max(np.max(y0), rms0)
    mu0 = float(np.argmax(y0))
    sigma0 = 4.0
    p0 = [amp0, mu0, sigma0, baseline0]

    bounds = (
        [0.0, 0.0, 0.5, -np.inf],
        [np.inf, nbin - 1, nbin / 4, np.inf],
    )

    def model(x, amp, mu, sigma, baseline):
        return gaussian_only_model_full(nbin, amp, mu, sigma, baseline)[x.astype(int)]

    sigma_data = np.full_like(y_fit, rms0, dtype=float)

    popt, pcov = curve_fit(
        model,
        x_fit,
        y_fit,
        p0=p0,
        bounds=bounds,
        sigma=sigma_data,
        absolute_sigma=True,
        maxfev=100000,
    )
    perr = np.sqrt(np.diag(pcov))

    model_full = gaussian_only_model_full(nbin, *popt)
    resid_fit = y_fit - model(x_fit, *popt)
    resid_full = y - model_full

    chi2 = np.sum((resid_fit / rms0) ** 2)

    return {
        "model_name": "gaussian_only",
        "params": popt,
        "errors": perr,
        "model_full": model_full,
        "residuals_full": resid_full,
        "fit_idx": fit_idx,
        "rms": rms0,
        "chi2_red": reduced_chi2(resid_fit, rms0, len(popt)),
        "aic": aic(chi2, len(popt)),
        "bic": bic(chi2, len(popt), len(y_fit)),
    }


def fit_scattered_gaussian(profile: np.ndarray) -> dict:
    y = np.asarray(profile, dtype=float)
    nbin = len(y)

    baseline0, rms0 = estimate_baseline_and_rms(y)
    y0 = y - baseline0

    fit_idx = find_fit_window(y, nsigma=3.0, pad=15)
    x_fit = fit_idx.astype(float)
    y_fit = y[fit_idx]

    amp0 = max(np.max(y0), rms0)
    mu0 = float(np.argmax(y0))

    sigma_grid = [2.0, 4.0, 6.0]
    tau_grid = [5.0, 10.0, 20.0, 40.0, 80.0]

    best = None

    def model(x, amp, mu, sigma, tau, baseline):
        return scattered_gaussian_model_full(nbin, amp, mu, sigma, tau, baseline)[x.astype(int)]

    sigma_data = np.full_like(y_fit, rms0, dtype=float)
    bounds = (
        [0.0, 0.0, 0.5, 0.2, -np.inf],
        [np.inf, nbin - 1, nbin / 4, nbin, np.inf],
    )

    for sigma0 in sigma_grid:
        for tau0 in tau_grid:
            p0 = [amp0, mu0, sigma0, tau0, baseline0]
            try:
                popt, pcov = curve_fit(
                    model,
                    x_fit,
                    y_fit,
                    p0=p0,
                    bounds=bounds,
                    sigma=sigma_data,
                    absolute_sigma=True,
                    maxfev=100000,
                )
                perr = np.sqrt(np.diag(pcov))
                resid_fit = y_fit - model(x_fit, *popt)
                chi2_red_val = reduced_chi2(resid_fit, rms0, len(popt))
                chi2 = np.sum((resid_fit / rms0) ** 2)

                candidate = {
                    "popt": popt,
                    "perr": perr,
                    "chi2_red": chi2_red_val,
                    "chi2": chi2,
                }

                if best is None or candidate["chi2_red"] < best["chi2_red"]:
                    best = candidate

            except Exception:
                continue

    if best is None:
        raise RuntimeError("Scattered fit failed for all initial guesses")

    popt = best["popt"]
    perr = best["perr"]
    model_full = scattered_gaussian_model_full(nbin, *popt)
    resid_fit = y_fit - model(x_fit, *popt)
    resid_full = y - model_full

    return {
        "model_name": "scattered_gaussian",
        "params": popt,
        "errors": perr,
        "model_full": model_full,
        "residuals_full": resid_full,
        "fit_idx": fit_idx,
        "rms": rms0,
        "chi2_red": best["chi2_red"],
        "aic": aic(best["chi2"], len(popt)),
        "bic": bic(best["chi2"], len(popt), len(y_fit)),
    }
