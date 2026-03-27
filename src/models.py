import numpy as np


def gaussian_profile(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def circular_scattering_kernel(nbin: int, tau: float) -> np.ndarray:
    x = np.arange(nbin, dtype=float)
    k = np.exp(-x / tau)
    k /= np.sum(k)
    return k


def circular_convolve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)).real


def gaussian_only_model_full(
    nbin: int,
    amp: float,
    mu: float,
    sigma: float,
    baseline: float
) -> np.ndarray:
    x = np.arange(nbin, dtype=float)
    return amp * gaussian_profile(x, mu, sigma) + baseline


def scattered_gaussian_model_full(
    nbin: int,
    amp: float,
    mu: float,
    sigma: float,
    tau: float,
    baseline: float,
) -> np.ndarray:
    x = np.arange(nbin, dtype=float)
    intrinsic = gaussian_profile(x, mu, sigma)
    kernel = circular_scattering_kernel(nbin, tau)
    broadened = circular_convolve(intrinsic, kernel)
    return amp * broadened + baseline
