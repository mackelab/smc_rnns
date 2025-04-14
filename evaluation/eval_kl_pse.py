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
from vi_rnn.generate import generate


def eval_kl_pse(
    vae,
    task,
    cut_off=0,
    init_state_eval="posterior_sample",
    smoothing=20,
    freq_cut_off=10000,
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
    with torch.no_grad():

        data = task.data_eval
        u = task.stim_eval
        if len(task.data_eval.shape) == 2:
            data = data.unsqueeze(0)
            u = u.unsqueeze(0)
        dur = min(data.shape[2], 10000)
        data = data[:, :, :dur]
        _, data_gen, _ = generate(
            vae,
            u=u,
            x=data,
            dur=dur,
            initial_state=init_state_eval,
            cut_off=cut_off,
        )
        data = data.permute(0, 2, 1).reshape(-1, vae.dim_x)
        data_gen = data_gen[:,:,:,0].permute(0, 2, 1).reshape(-1, vae.dim_x)
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
