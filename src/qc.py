def classify_fit(
    snr: float,
    tau_bin: float | None,
    tau_bin_err: float | None,
    chi2_red: float,
    delta_bic: float,
) -> str:
    if snr < 5:
        return "LOW_SN"

    if tau_bin is None or tau_bin_err is None:
        return "BAD_FIT"

    if tau_bin <= 1.0:
        return "UPPER_LIMIT"

    if tau_bin_err <= 0 or (tau_bin / tau_bin_err) < 3:
        return "MARGINAL"

    if chi2_red > 3.0:
        return "BAD_FIT"

    if delta_bic > -6:
        return "NO_STRONG_SCATTERING"

    return "GOOD"
