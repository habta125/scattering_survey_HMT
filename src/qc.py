import numpy as np

def classify_fit(
    snr: float,
    tau_bin: float | None,
    tau_bin_err: float | None,
    chi2_red: float,
    delta_bic: float | None = None,
) -> str:

    if snr < 5:
        return "LOW_SN"

    if tau_bin is None:
        return "BAD_FIT"

    if tau_bin <= 1:
        return "UPPER_LIMIT"

    if chi2_red > 10:
        return "POOR_MODEL"

    if tau_bin_err is None or not np.isfinite(tau_bin_err) or tau_bin_err <= 0:
        if tau_bin >= 10:
            return "STRONG_SCATTERING"
        return "BAD_FIT"

    if tau_bin / tau_bin_err < 3:
        return "MARGINAL"

    if tau_bin < 10:
        return "WEAK_SCATTERING"

    return "STRONG_SCATTERING"
