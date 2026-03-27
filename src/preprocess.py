import numpy as np


def estimate_baseline_and_rms(
    profile: np.ndarray,
    offpulse_fraction: float = 0.25
) -> tuple[float, float]:
    n = len(profile)
    n_off = max(16, int(offpulse_fraction * n))
    offpulse = np.sort(profile)[:n_off]
    baseline = float(np.median(offpulse))
    rms = float(np.std(offpulse))
    return baseline, max(rms, 1e-12)


def subtract_baseline(profile: np.ndarray) -> tuple[np.ndarray, float, float]:
    baseline, rms = estimate_baseline_and_rms(profile)
    return profile - baseline, baseline, rms


def estimate_snr(profile: np.ndarray) -> float:
    y, _, rms = subtract_baseline(profile)
    return float(np.max(y) / rms)


def find_fit_window(
    profile: np.ndarray,
    nsigma: float = 3.0,
    pad: int = 15
) -> np.ndarray:
    y, _, rms = subtract_baseline(profile)
    above = np.where(y > nsigma * rms)[0]

    if len(above) == 0:
        return np.arange(len(profile))

    start = max(0, int(above[0]) - pad)
    end = min(len(profile), int(above[-1]) + pad + 1)
    return np.arange(start, end)
