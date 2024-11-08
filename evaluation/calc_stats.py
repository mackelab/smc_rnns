import numpy as np


def calculate_correlation(data):
    """Calculate the correlation matrix for a dataset.
    Args:
        data: np.array (T x N), dataset
    Returns:
        correlation_matrix: np.array (N x N), correlation matrix
    """
    correlation_matrix = np.corrcoef(data, rowvar=False)
    return correlation_matrix


def estimate_cross_correlation(data, lag, len):
    """Calculate the cross-correlation of a dataset.
    Args:
        data: np.array (T), dataset
        lag: int, lag for cross-correlation
        len: int, length of the dataset to go over
    Returns:
        Corrs: list of np.arrays (lag), cross-correlations
    """
    Corrs = []
    for i in range(0, len, lag):
        Corr = np.flip(
            np.correlate(data[i : i + lag], data[i : i + lag * 2], mode="valid")
        )
        Corr /= Corr[0]
        Corrs.append(Corr)
    return Corrs


def calc_isi_stats(spikes, dt=1):
    """Calculate the CV and mean of the ISI per neuron for a dataset
    Args:
        spikes: np.array (T x N), spike trains
    Returns:
        CVs_isi: list of floats (N), coefficient of variation of the ISI
        Means_isi: list of floats (N), mean of the ISI

    """
    CVs_isi = []
    Means_isi = []
    Std_isi = []
    for spike_trace in spikes.T:
        isi = np.diff(np.where(spike_trace)[0] * dt)
        CVs_isi.append(np.std(isi) / np.mean(isi))
        Means_isi.append(np.mean(isi))
        Std_isi.append(np.std(isi))
    return CVs_isi, Means_isi, Std_isi


def calculate_correlation(data):
    """Calculate the correlation matrix for a dataset.
    Args:
        data: np.array (T x N), dataset
    Returns:
        correlation_matrix: np.array (N x N), correlation matrix
    """
    correlation_matrix = np.corrcoef(data, rowvar=False)
    return correlation_matrix


def calc_isi_stats_per_trial(spikes, dt=1, exclude_thr=1):
    """Calculate the CV and mean of the ISI per neuron for a dataset
        with trial structure (Bs trials)
    Args:
        spikes: np.array (Bs x T x N), spike trains
    Returns:
        CVs_isi: list of floats (N), coefficient of variation of the ISI
        Means_isi: list of floats (N), mean of the ISI

    """

    n_trials, n_timesteps, n_neurons = spikes.shape

    CVs_isi = []
    Means_isi = []
    Std_isi = []

    for i in range(n_neurons):
        spike_traces = spikes[:, :, i]
        isi = np.concatenate(
            [np.diff(np.where(spike_trace)[0] * dt) for spike_trace in spike_traces]
        )
        std = np.std(isi)
        mean = np.mean(isi)
        if len(isi) < exclude_thr or std == 0:
            print("setting to nan, neuron " + str(i))
            CVs_isi.append(np.nan)
            Means_isi.append(np.nan)
            Std_isi.append(np.nan)
        else:
            CVs_isi.append(std / mean)
            Means_isi.append(mean)
            Std_isi.append(std)
    return np.array(CVs_isi), np.array(Means_isi), np.array(Std_isi)
