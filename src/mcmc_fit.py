import numpy as np
import emcee

from .fitter import fit_scattering_1comp_fast, fit_scattering_2comp_fast
from .models import one_comp_fit, one_comp_full, two_comp_fit, two_comp_full
from .qc import classify_result, compute_metrics
from .utils import estimate_baseline_and_rms, estimate_snr


def log_prior_1c(theta, nbin):
    amp, mu, sigma, tau, baseline = theta
    if not (0.0 < amp < 3.0):
        return -np.inf
    if not (0.0 <= mu <= nbin - 1):
        return -np.inf
    if not (0.3 < sigma < nbin / 4.0):
        return -np.inf
    if not (0.1 < tau < nbin / 2.0):
        return -np.inf
    if not (-2.0 < baseline < 2.0):
        return -np.inf
    return -np.log(sigma) - np.log(tau)


def log_likelihood_1c(theta, x_fit, y_fit, yerr, nbin):
    model = one_comp_fit(x_fit, *theta, nbin)
    if not np.all(np.isfinite(model)):
        return -np.inf
    inv_sigma2 = 1.0 / (yerr ** 2)
    return -0.5 * np.sum((y_fit - model) ** 2 * inv_sigma2 + np.log(2.0 * np.pi * yerr ** 2))


def log_probability_1c(theta, x_fit, y_fit, yerr, nbin):
    lp = log_prior_1c(theta, nbin)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood_1c(theta, x_fit, y_fit, yerr, nbin)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


def log_prior_2c(theta, nbin):
    amp1, mu1, sigma1, amp2, mu2, sigma2, tau, baseline = theta
    if not (0.0 < amp1 < 3.0 and 0.0 < amp2 < 3.0):
        return -np.inf
    if not (0.0 <= mu1 <= nbin - 1 and 0.0 <= mu2 <= nbin - 1):
        return -np.inf
    if not (0.2 < sigma1 < nbin / 4.0 and 0.2 < sigma2 < nbin / 4.0):
        return -np.inf
    if not (0.1 < tau < nbin / 2.0):
        return -np.inf
    if not (-2.0 < baseline < 2.0):
        return -np.inf
    return -np.log(sigma1) - np.log(sigma2) - np.log(tau)


def log_likelihood_2c(theta, x_fit, y_fit, yerr, nbin):
    model = two_comp_fit(x_fit, *theta, nbin)
    if not np.all(np.isfinite(model)):
        return -np.inf
    inv_sigma2 = 1.0 / (yerr ** 2)
    return -0.5 * np.sum((y_fit - model) ** 2 * inv_sigma2 + np.log(2.0 * np.pi * yerr ** 2))


def log_probability_2c(theta, x_fit, y_fit, yerr, nbin):
    lp = log_prior_2c(theta, nbin)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood_2c(theta, x_fit, y_fit, yerr, nbin)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


def _init_walkers(center, scales, nwalkers, log_prior_fn, nbin, rng):
    pos = []
    trials = 0
    while len(pos) < nwalkers and trials < 100000:
        trial = center + rng.normal(0.0, scales, size=len(center))
        if np.isfinite(log_prior_fn(trial, nbin)):
            pos.append(trial)
        trials += 1
    if len(pos) < nwalkers:
        raise RuntimeError("Could not initialize enough valid MCMC walkers.")
    return np.asarray(pos)


def fit_scattering_1comp_mcmc(profile, period_sec, nwalkers=32, nsteps=2500, burn_frac=0.4, thin=10, random_seed=1234):
    rng = np.random.default_rng(random_seed)
    y = np.asarray(profile, dtype=float)
    nbin = len(y)
    fast = fit_scattering_1comp_fast(y, period_sec)
    fit_idx = fast["fit_idx"]
    x_fit = fit_idx.astype(float)
    y_fit = y[fit_idx]
    _, rms0 = estimate_baseline_and_rms(y)
    yerr = np.full_like(y_fit, rms0, dtype=float)

    center = np.array([fast["amp"], fast["mu"], fast["sigma_bin"], fast["tau_bin"], fast["baseline"]], dtype=float)
    scales = np.array([0.05 * max(center[0], 1e-3), 0.02 * nbin, 0.10 * max(center[2], 1.0), 0.10 * max(center[3], 1.0), 0.05], dtype=float)
    pos = _init_walkers(center, scales, nwalkers, log_prior_1c, nbin, rng)

    sampler = emcee.EnsembleSampler(nwalkers, 5, log_probability_1c, args=(x_fit, y_fit, yerr, nbin))
    sampler.run_mcmc(pos, nsteps, progress=False)

    burn = int(burn_frac * nsteps)
    flat_samples = sampler.get_chain(discard=burn, thin=thin, flat=True)
    flat_log_prob = sampler.get_log_prob(discard=burn, thin=thin, flat=True)
    if flat_samples.shape[0] < 50:
        raise RuntimeError("Too few post-burn MCMC samples for 1-component fit.")

    p16 = np.percentile(flat_samples, 16, axis=0)
    p50 = np.percentile(flat_samples, 50, axis=0)
    p84 = np.percentile(flat_samples, 84, axis=0)
    best_theta = p50
    model_full = one_comp_full(nbin, *best_theta)
    y_model_fit = model_full[fit_idx]
    chi2, chi2_red, aic, bic = compute_metrics(y_fit, y_model_fit, rms0, 5)
    bin_width_ms = (period_sec * 1000.0) / nbin

    result = {
        "model_name": "1-component-MCMC",
        "selected_model": "1comp",
        "n_components": 1,
        "amp": p50[0], "amp_err": 0.5 * (p84[0] - p16[0]), "amp_p16": p16[0], "amp_p50": p50[0], "amp_p84": p84[0],
        "mu": p50[1], "mu_err": 0.5 * (p84[1] - p16[1]), "mu_p16": p16[1], "mu_p50": p50[1], "mu_p84": p84[1],
        "sigma_bin": p50[2], "sigma_bin_err": 0.5 * (p84[2] - p16[2]), "sigma_bin_p16": p16[2], "sigma_bin_p50": p50[2], "sigma_bin_p84": p84[2],
        "tau_bin": p50[3], "tau_bin_err": 0.5 * (p84[3] - p16[3]), "tau_bin_p16": p16[3], "tau_bin_p50": p50[3], "tau_bin_p84": p84[3],
        "baseline": p50[4], "baseline_err": 0.5 * (p84[4] - p16[4]), "baseline_p16": p16[4], "baseline_p50": p50[4], "baseline_p84": p84[4],
        "tau_ms": p50[3] * bin_width_ms, "tau_ms_err": 0.5 * (p84[3] - p16[3]) * bin_width_ms,
        "tau_ms_p16": p16[3] * bin_width_ms, "tau_ms_p50": p50[3] * bin_width_ms, "tau_ms_p84": p84[3] * bin_width_ms,
        "sigma_ms": p50[2] * bin_width_ms, "sigma_ms_err": 0.5 * (p84[2] - p16[2]) * bin_width_ms,
        "sigma_ms_p16": p16[2] * bin_width_ms, "sigma_ms_p50": p50[2] * bin_width_ms, "sigma_ms_p84": p84[2] * bin_width_ms,
        "snr": estimate_snr(y), "chi2": chi2, "chi2_red": chi2_red, "aic": aic, "bic": bic,
        "fit_idx": fit_idx, "model_full": model_full, "residuals_full": y - model_full,
        "period_sec": period_sec, "nbin": nbin, "bin_width_ms": bin_width_ms,
        "mcmc_nwalkers": nwalkers, "mcmc_nsteps": nsteps, "mcmc_burn": burn, "mcmc_thin": thin,
        "mcmc_nsamples": flat_samples.shape[0], "mcmc_chain": sampler.get_chain(),
        "flat_samples": flat_samples, "logpost_max": float(np.max(flat_log_prob)),
        "param_labels": ["amp", "mu", "sigma_bin", "tau_bin", "baseline"],
    }
    result["classification"] = classify_result(result)
    return result


def fit_scattering_2comp_mcmc(profile, period_sec, nwalkers=48, nsteps=3000, burn_frac=0.4, thin=10, random_seed=1234):
    rng = np.random.default_rng(random_seed)
    y = np.asarray(profile, dtype=float)
    nbin = len(y)
    fast = fit_scattering_2comp_fast(y, period_sec)
    fit_idx = fast["fit_idx"]
    x_fit = fit_idx.astype(float)
    y_fit = y[fit_idx]
    _, rms0 = estimate_baseline_and_rms(y)
    yerr = np.full_like(y_fit, rms0, dtype=float)

    center = np.array([
        fast["amp1"], fast["mu1"], fast["sigma1_bin"],
        fast["amp2"], fast["mu2"], fast["sigma2_bin"],
        fast["tau_bin"], fast["baseline"]
    ], dtype=float)
    scales = np.array([
        0.05 * max(center[0], 1e-3), 0.02 * nbin, 0.10 * max(center[2], 1.0),
        0.05 * max(center[3], 1e-3), 0.02 * nbin, 0.10 * max(center[5], 1.0),
        0.10 * max(center[6], 1.0), 0.05
    ], dtype=float)
    pos = _init_walkers(center, scales, nwalkers, log_prior_2c, nbin, rng)

    sampler = emcee.EnsembleSampler(nwalkers, 8, log_probability_2c, args=(x_fit, y_fit, yerr, nbin))
    sampler.run_mcmc(pos, nsteps, progress=False)

    burn = int(burn_frac * nsteps)
    flat_samples = sampler.get_chain(discard=burn, thin=thin, flat=True)
    flat_log_prob = sampler.get_log_prob(discard=burn, thin=thin, flat=True)
    if flat_samples.shape[0] < 50:
        raise RuntimeError("Too few post-burn MCMC samples for 2-component fit.")

    p16 = np.percentile(flat_samples, 16, axis=0)
    p50 = np.percentile(flat_samples, 50, axis=0)
    p84 = np.percentile(flat_samples, 84, axis=0)
    best_theta = p50
    model_full = two_comp_full(nbin, *best_theta)
    y_model_fit = model_full[fit_idx]
    chi2, chi2_red, aic, bic = compute_metrics(y_fit, y_model_fit, rms0, 8)
    bin_width_ms = (period_sec * 1000.0) / nbin

    result = {
        "model_name": "2-component-MCMC",
        "selected_model": "2comp",
        "n_components": 2,
        "amp1": p50[0], "amp1_err": 0.5 * (p84[0] - p16[0]), "amp1_p16": p16[0], "amp1_p50": p50[0], "amp1_p84": p84[0],
        "mu1": p50[1], "mu1_err": 0.5 * (p84[1] - p16[1]), "mu1_p16": p16[1], "mu1_p50": p50[1], "mu1_p84": p84[1],
        "sigma1_bin": p50[2], "sigma1_bin_err": 0.5 * (p84[2] - p16[2]), "sigma1_bin_p16": p16[2], "sigma1_bin_p50": p50[2], "sigma1_bin_p84": p84[2],
        "amp2": p50[3], "amp2_err": 0.5 * (p84[3] - p16[3]), "amp2_p16": p16[3], "amp2_p50": p50[3], "amp2_p84": p84[3],
        "mu2": p50[4], "mu2_err": 0.5 * (p84[4] - p16[4]), "mu2_p16": p16[4], "mu2_p50": p50[4], "mu2_p84": p84[4],
        "sigma2_bin": p50[5], "sigma2_bin_err": 0.5 * (p84[5] - p16[5]), "sigma2_bin_p16": p16[5], "sigma2_bin_p50": p50[5], "sigma2_bin_p84": p84[5],
        "tau_bin": p50[6], "tau_bin_err": 0.5 * (p84[6] - p16[6]), "tau_bin_p16": p16[6], "tau_bin_p50": p50[6], "tau_bin_p84": p84[6],
        "baseline": p50[7], "baseline_err": 0.5 * (p84[7] - p16[7]), "baseline_p16": p16[7], "baseline_p50": p50[7], "baseline_p84": p84[7],
        "tau_ms": p50[6] * bin_width_ms, "tau_ms_err": 0.5 * (p84[6] - p16[6]) * bin_width_ms,
        "tau_ms_p16": p16[6] * bin_width_ms, "tau_ms_p50": p50[6] * bin_width_ms, "tau_ms_p84": p84[6] * bin_width_ms,
        "sigma1_ms": p50[2] * bin_width_ms, "sigma1_ms_err": 0.5 * (p84[2] - p16[2]) * bin_width_ms,
        "sigma1_ms_p16": p16[2] * bin_width_ms, "sigma1_ms_p50": p50[2] * bin_width_ms, "sigma1_ms_p84": p84[2] * bin_width_ms,
        "sigma2_ms": p50[5] * bin_width_ms, "sigma2_ms_err": 0.5 * (p84[5] - p16[5]) * bin_width_ms,
        "sigma2_ms_p16": p16[5] * bin_width_ms, "sigma2_ms_p50": p50[5] * bin_width_ms, "sigma2_ms_p84": p84[5] * bin_width_ms,
        "snr": estimate_snr(y), "chi2": chi2, "chi2_red": chi2_red, "aic": aic, "bic": bic,
        "fit_idx": fit_idx, "model_full": model_full, "residuals_full": y - model_full,
        "period_sec": period_sec, "nbin": nbin, "bin_width_ms": bin_width_ms,
        "mcmc_nwalkers": nwalkers, "mcmc_nsteps": nsteps, "mcmc_burn": burn, "mcmc_thin": thin,
        "mcmc_nsamples": flat_samples.shape[0], "mcmc_chain": sampler.get_chain(),
        "flat_samples": flat_samples, "logpost_max": float(np.max(flat_log_prob)),
        "param_labels": ["amp1", "mu1", "sigma1_bin", "amp2", "mu2", "sigma2_bin", "tau_bin", "baseline"],
    }
    result["classification"] = classify_result(result)
    return result
