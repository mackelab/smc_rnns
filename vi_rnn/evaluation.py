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
from vi_rnn.initialize_parameterize import chol_cov_embed


def get_initial_state(vae,u,x=None,initial_state="prior_sample",optimal_proposal=False):

    if x is not None:
        x = x
    u = u
    with torch.no_grad():

        #get prior mean and std
        prior_mean = vae.rnn.get_initial_state(u[:,:,0]).unsqueeze(2) # Bs, Dz,1
   
      
        if initial_state=="prior_sample":
            if vae.rnn.params["noise_z"]=="full":
                chol_prior_t0 = chol_cov_embed(vae.rnn.R_z_t0)
                Q_dist = torch.distributions.MultivariateNormal(loc=prior_mean.squeeze(-1), scale_tril=chol_prior_t0.unsqueeze(0))
                z0 = Q_dist.sample().unsqueeze(-1)
            else:
                eff_var_prior_t0 = torch.clamp(
                    vae.rnn.var_embed_z_t0(vae.rnn.R_z_t0).unsqueeze(0).unsqueeze(-1),
                    min=vae.min_var,
                    max=vae.max_var,
                )  # 1,Dz,1
                Q_dist = torch.distributions.Normal(loc=prior_mean, scale=torch.sqrt(eff_var_prior_t0))
                z0 = Q_dist.sample()
        elif initial_state=="prior_mean":
            z0 = prior_mean

        elif optimal_proposal:  
            eff_var_x = torch.clip(vae.rnn.var_embed_x(vae.rnn.R_x), 1e-8)
            eff_var_x_inv = 1.0 / torch.clip(vae.rnn.var_embed_x(vae.rnn.R_x), 1e-8)

            eff_var_prior_t0_chol = chol_cov_embed(vae.rnn.R_z_t0)
            # Get the observation weights and bias
            if vae.rnn.params["readout_from"] == "currents":
                m = vae.rnn.transition.m
                B = vae.rnn.observation.cast_B(vae.rnn.observation.B).T @ m
                B = B.T
            else:
                B = vae.rnn.observation.cast_B(vae.rnn.observation.B)
            Obs_bias = vae.rnn.observation.Bias.squeeze(-1)

            # Calculate the Kalman gain and interpolation alpha
            if vae.dim_x < vae.dim_z*4:
                Kalman_gain = (
                    eff_var_prior_t0
                    @ B
                    @ torch.linalg.inv(torch.diag(eff_var_x) + B.T @ eff_var_prior_t0 @ B)
                )
                alpha = Kalman_gain @ B.T
                one_min_alpha = torch.eye(vae.dim_z, device=alpha.device) - alpha

                # Posterior Joseph stabilised Covariance
                var_Q = (
                    one_min_alpha @ eff_var_prior_t0 @ one_min_alpha.T
                    + (Kalman_gain * torch.unsqueeze(eff_var_x, 0)) @ Kalman_gain.T
                )

            else:  # this is generally faster for low-rank models
                var_Q = torch.linalg.inv(
                    torch.cholesky_inverse(eff_var_prior_t0_chol)
                    + (B * torch.unsqueeze(eff_var_x_inv, 0)) @ B.T
                )
                Kalman_gain = var_Q @ (B * torch.unsqueeze(eff_var_x_inv, 0))
                alpha = Kalman_gain @ B.T
                one_min_alpha = torch.eye(vae.dim_z, device=alpha.device) - alpha

            # avoid numerical issues
            var_Q = (
                torch.eye(vae.dim_z, device=alpha.device) * 1e-8 + (var_Q + var_Q.T) / 2
            )

            var_Q_cholesky = torch.linalg.cholesky(var_Q)
            # Posterior Mean

            mean_Q = torch.einsum("zs,BsK->BzK", one_min_alpha, prior_mean) + torch.einsum(
                "zx,BxK->BzK", Kalman_gain, x[:, :, 0].unsqueeze(-1) - Obs_bias
            )

            if initial_state =="posterior_sample":
                Q_dist = torch.distributions.MultivariateNormal(
                loc=mean_Q.permute(0, 2, 1), scale_tril=var_Q_cholesky
                )
                z0 = Q_dist.sample()

            elif initial_state =="posterior_mean":
                z0 = mean_Q


            else:
                raise ValueError("initial state not recognized, use prior_sample, prior_mean, posterior_sample or posterior_mean")

        else:# we include the encoder
            x_enc = torch.clone(x)

            Emean, log_Evar = vae.encoder(x_enc)


            Evar = torch.clamp(torch.exp(log_Evar[:,:,0]), min=vae.min_var, max=vae.max_var)  # Bs,Dx,T,K
             
            precZ = 1 / eff_var_prior_t0
            precE = 1 / Evar
            precQ = precZ + precE
            alpha = precE / precQ
            mean_Q = (1 - alpha) * prior_mean + alpha * Emean[:, :, 0]

            if initial_state =="posterior_sample":
                eff_var_Q = 1 / precQ
                Q_dist = torch.distributions.Normal(loc=mean_Q, scale=torch.sqrt(eff_var_Q))
                z0 = Q_dist.sample()

            elif initial_state=="posterior_mean":
                z0 = mean_Q
     
            else:
                raise ValueError("initial state not recognized, use prior_sample, prior_mean, posterior_sample or posterior_mean")
                
    return z0


def predict(vae,u=None,x=None,initial_state="prior_sample", observation_model="Gauss",optimal_proposal=False,
            verbose=True, sim_v=True):
    with torch.no_grad():
        if u is None:
            u = torch.zeros(x.shape[0],0,x.shape[2])

        if isinstance(initial_state, str):
            if verbose:
                print("initial state is " + initial_state)
            with torch.no_grad():
                z0 = get_initial_state(vae,u,x,initial_state=initial_state,optimal_proposal=optimal_proposal)
        else:
            z0 = initial_state
        dur = u.shape[2]

    
        Z, v = vae.rnn.get_latent_time_series(time_steps=dur, z0=z0, u=u, noise_scale=1,sim_v=sim_v)
        
        rates = (
            vae.rnn.get_observation(Z, v=v,noise_scale=0)
        ).cpu().detach().numpy()[:, :, :, 0]
        
        if observation_model=="Gauss":
            data_gen = (
                vae.rnn.get_observation(Z, v=v,noise_scale=1)
            ).cpu().detach().numpy()[:, :, :, 0]

        elif observation_model == "Poisson":
            data_gen = np.random.poisson(rates).astype('float64')
        Z = Z.cpu().detach().numpy()[:,:,:,0]
   
    return Z, data_gen, rates



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
            dim_x, T = data.shape
            if sim_latent_noise > 1e-8:  # take sample of encoder
                z_hat, _, _, _ = vae.encoder(data[:,:trial_dur].unsqueeze(0))
            else:  # take mean prediction of encoder
                _, z_hat, _, _ = vae.encoder(data[:,:trial_dur].unsqueeze(0))
            z0 = z_hat[:, :, :1,0]
            Z = vae.rnn.get_latent_time_series(
                time_steps=T, cut_off=cut_off, z0=z0, noise_scale=sim_latent_noise
            )
            data = data.T
        # Evaluate on multiple short trajectories (trials)
        else:
            max_trials, dim_x, T_data_trial = task.data_eval.shape
            n_trials = int(10000 / T_data_trial)
            n_eval_trials = min(n_trials, max_trials)
            data = task.data_eval[
                :n_eval_trials
            ]  
            z_hat, _,  = vae.encoder(data)

            z0 = z_hat[:, :, :1]
            Z = vae.rnn.get_latent_time_series(
                time_steps=T_data_trial,
                cut_off=cut_off,
                z0=z0,
                noise_scale=sim_latent_noise,
            )

            data = data.permute(0, 2, 1).reshape(-1, dim_x)
            T, dim_x = data.shape

        data_gen = (
            vae.rnn.get_observation(Z, noise_scale=sim_obs_noise)[:,:,:,0]
            .permute(0, 2, 1)
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

