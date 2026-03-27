import numpy as np


def reduced_chi2(residuals: np.ndarray, rms: float, n_params: int) -> float:
    dof = max(1, len(residuals) - n_params)
    return float(np.sum((residuals / max(rms, 1e-12)) ** 2) / dof)


def aic(chi2: float, k: int) -> float:
    return float(chi2 + 2 * k)


def bic(chi2: float, k: int, n: int) -> float:
    return float(chi2 + k * np.log(max(n, 1)))
