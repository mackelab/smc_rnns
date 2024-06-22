import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import zscore

# ADAPTED FROM:
# https://github.com/DurstewitzLab/ChaosRNN
# https://github.com/DurstewitzLab/dendPLRNN
# https://github.com/DurstewitzLab/GTF-shPLRNN


def ensure_length_is_even(x):
    """Ensure that the length of the input is even"""
    n = len(x)
    if n % 2 != 0:
        x = x[:-1]
        n = len(x)
    x = np.reshape(x, (1, n))
    return x


def fft_smoothed(x, smoothing):
    """
    Compute the smoothed power spectrum of a 1D signal
    Args:
        x (np.ndarray): 1D array
        smoothing (float): smoothing parameter for the power spectrum
    Returns:
        fft_smoothed (np.ndarray): normalised and smoothed power spectrum
    """
    eps = 1e-8

    x = ensure_length_is_even(x)
    fft_real = np.fft.rfft(x, norm="ortho")
    fft_magnitude = np.abs(fft_real) ** 2 * 2 / len(x)
    fft_smoothed = kernel_smoothen(fft_magnitude, kernel_sigma=smoothing)
    fft_smoothed[fft_smoothed < 0] = 0
    return fft_smoothed / (np.sum(fft_smoothed) + eps)


def get_average_spectrum(trajectories, smoothing):
    """
    Get the average power spectrum of a set of trajectories
    Args:
        trajectories (np.ndarray): set of trajectories
        smoothing (float): smoothing parameter for the power spectrum
    Returns:
        spectrum (np.ndarray): average power spectrum
    """
    spectrum = []
    for trajectory in trajectories:
        trajectory = zscore(trajectory)
        fft = fft_smoothed(trajectory, smoothing)
        spectrum.append(fft)
    spectrum = np.nanmean(np.array(spectrum), axis=0)
    return spectrum


def power_spectrum_helling_per_dim(x_gen, x_true, smoothing, freq_cutoff):
    """
    Compute helling distance per data dimension
    Args:
        x_gen: generated data
        x_true: true data
        smoothing: smoothing parameter for the power spectrum
        freq_cutoff: cut off for the power spectrum
    Returns:
        pse_corrs_per_dim: helling distance per data dimension

    """
    assert x_true.shape[1] == x_gen.shape[1]
    assert x_true.shape[2] == x_gen.shape[2]
    dim_x = x_gen.shape[2]
    pse_corrs_per_dim = []
    for dim in range(dim_x):
        spectrum_true = get_average_spectrum(x_true[:, :, dim], smoothing)
        spectrum_gen = get_average_spectrum(x_gen[:, :, dim], smoothing)
        spectrum_true = spectrum_true[:, :freq_cutoff]
        spectrum_gen = spectrum_gen[:, :freq_cutoff]
        spectrum_true /= np.sum(spectrum_true)
        spectrum_gen /= np.sum(spectrum_gen)
        hellinger_dist = (1 / np.sqrt(2)) * np.sqrt(
            np.sum((np.sqrt(spectrum_gen) - np.sqrt(spectrum_true)) ** 2)
        )
        pse_corrs_per_dim.append(hellinger_dist)
    return pse_corrs_per_dim


def power_spectrum_helling(x_gen, x_true, smoothing, freq_cutoff):
    """
    Compute mean helling distance over data dimensions
    Args:
        x_gen: generated data
        x_true: true data
        smoothing: smoothing parameter for the power spectrum
        freq_cutoff: cut off for the power spectrum
    Returns:
        pse_corrs_per_dim: mean helling distance over data dimensions

    """
    pse_errors_per_dim = power_spectrum_helling_per_dim(
        x_gen, x_true, smoothing, freq_cutoff
    )
    return np.array(pse_errors_per_dim).mean(axis=0)


def kernel_smoothen(data, kernel_sigma=1):
    """
    Smoothen data with Gaussian kernel
    Args:
        data: data to be smoothened
        kernel_sigma: width of Gaussian kernel
    Returns:
        data: smoothened data
    """
    data = gaussian_filter1d(data, kernel_sigma)
    return data
