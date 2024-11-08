import os
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
from evaluation.kl_Gauss import calc_kl_from_data
from evaluation.pse import power_spectrum_helling
import torch
import scipy.ndimage as ndimage
import scipy.signal as signal
import numpy as np
from scipy.stats import zscore


def eval_VAE(
    vae,
    task,
    smoothing=20,
    cut_off=0,
    freq_cut_off=10000,
    sim_obs_noise=1,
    sim_latent_noise=1,
    smooth_at_eval=True,
):
    """
    Evaluate the VAE by looking at distribution over states and time

    Args:
        vae (nn.Module): VAE model
        task (Basic_dataset): dataset object
        smoothing (int): smoothing parameter for the power spectrum
        cut_off (int): cut off for the latent time series
        freq_cut_off (int): cut off for the power spectrum
        sim_obs_noise (float): observation noise scale
        sim_latent_noise (float): latent noise scale
        smooth_at_eval (bool): whether to smooth the generated data at evaluation time

    Returns:
        klx_bin (float): KL divergence between the true and generated data
        psH (float): power spectrum distance between the true and generated data
        mean_rate_error (float): mean rate error between the true and generated data

    """
    trial_data, _ = task.__getitem__(0)
    trial_dur = trial_data.shape[1]
    with torch.no_grad():

        # Evaluate on one long trajectory
        if len(task.data_eval.shape) == 2:
            data = task.data_eval
            T, dim_x = data.shape
            if sim_latent_noise > 1e-8:  # take sample of encoder
                z_hat, _, _, _ = vae.encoder(data[:trial_dur].T.unsqueeze(0))
            else:  # take mean prediction of encoder
                _, z_hat, _, _ = vae.encoder(data[:trial_dur].T.unsqueeze(0))
            z0 = z_hat[:, :, :1].squeeze()
            Z = vae.rnn.get_latent_time_series(
                time_steps=T, cut_off=cut_off, z0=z0, noise_scale=sim_latent_noise
            )
        # Evaluate on multiple short trajectories (trials)
        else:
            dim_x, max_trials, T_data_trial = task.data_eval.shape
            n_trials = int(10000 / T_data_trial)
            n_eval_trials = min(n_trials, max_trials)
            data = task.data_eval[
                :, :n_eval_trials, :
            ]  # dim_x, n_eval_trials, T_data_trial
            if sim_latent_noise > 1e-8:
                z_hat, _, _, _ = vae.encoder(data.permute(1, 0, 2))
            else:
                _, z_hat, _, _ = vae.encoder(data.permute(1, 0, 2))

            z0 = z_hat[:, :, :1]
            Z = vae.rnn.get_latent_time_series(
                time_steps=T_data_trial,
                cut_off=cut_off,
                z0=z0,
                noise_scale=sim_latent_noise,
            )

            data = data.reshape(dim_x, -1).T  # n_eval_trials*T_data_trial, dim_x
            T, dim_x = data.shape

        data_gen = (
            vae.rnn.get_observation(Z, noise_scale=sim_obs_noise)
            .permute(0, 2, 1, 3)
            .reshape(T, dim_x)
        )

        # potentially smooth
        if smooth_at_eval:
            window = signal.windows.hann(15)
            data_gen = torch.from_numpy(
                zscore(
                    ndimage.convolve1d(data_gen.cpu().numpy(), window, axis=0), axis=0
                )
            )

        klx_bin = calc_kl_from_data(data_gen, data.to(device=data_gen.device))

        # Helling distance accross time
        data = np.expand_dims(data.cpu().numpy(), 0)
        data_gen = np.expand_dims(data_gen.cpu().numpy(), 0)
        psH = power_spectrum_helling(
            data_gen, data, smoothing=smoothing, freq_cutoff=freq_cut_off
        )
        mean_rate_error = mean_rate(data_gen, data)
        print(
            f"KL_x = {klx_bin.item():.3f}, PS_dist = {psH:.3f}, Mean_rate_error = {mean_rate_error:.3f}"
        )
        return klx_bin.item(), psH, mean_rate_error


def mean_rate(data_gen, data_real):
    """Calculate the mean rate error between the true and generated data"""
    data_mean_rates = np.sum(data_real, axis=1)
    data_gen_mean_rates = np.sum(data_gen, axis=1)
    mean_rate_error = (
        np.mean((data_mean_rates - data_gen_mean_rates) ** 2) / data_gen.shape[1]
    )
    return mean_rate_error
