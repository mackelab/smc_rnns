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



def get_initial_state(vae,u,x=None,initial_state="prior_sample",max_trials=1000,normalize_data=False):

    if x is not None:
        x = x[:max_trials]
    u = u[:max_trials]
    if vae.rnn.params["noise_z"]=="full":
        full_cov = True
    else:
        full_cov = False

    with torch.no_grad():

        #get prior mean and std
        prior_mean = vae.rnn.get_initial_state(u[:,:,0]).unsqueeze(2) # Bs, Dz,1
        eff_var_prior_t0 = torch.clamp(
            vae.rnn.var_embed_z_t0(vae.rnn.R_z_t0).unsqueeze(0).unsqueeze(-1),
            min=vae.min_var,
            max=vae.max_var,
        )  # 1,Dz,1

      
        if initial_state=="prior_sample":
            if full_cov:
                chol_prior_t0 = vae.rnn.chol_cov_embed(vae.rnn.R_z_t0)
                Q_dist = torch.distributions.MultivariateNormal(loc=prior_mean.squeeze(-1), scale_tril=chol_prior_t0.unsqueeze(0))
                z0 = Q_dist.sample().unsqueeze(-1)
            else:
                Q_dist = torch.distributions.Normal(loc=prior_mean, scale=torch.sqrt(eff_var_prior_t0))
                z0 = Q_dist.sample()

        elif initial_state=="prior_mean":
            z0 = prior_mean

        else: # we include the encoder
            x_enc = torch.clone(x)
            if normalize_data:
                print("normalize data for enc")
                x_enc-=x.mean()
                x_enc/=x_enc.std()
            #print(x_enc.mean(),x_enc.std())
            Emean, log_Evar = vae.encoder(x_enc)
            if full_cov:
                E_chol = vae.rnn.chol_cov_embed(log_Evar[:,:,:,0]) #n_trials x dim_z x dim_z 
                E_prec = torch.cholesky_inverse(E_chol) #n_trials x dim_z x dim_z 
            else:
                Evar = torch.clamp(torch.exp(log_Evar[:,:,0]), min=vae.min_var, max=vae.max_var)  # Bs,Dx,T,K
            
            if initial_state=="encoder_mean":
                z0 = Emean[:, :, 0]
            
            elif initial_state=="encoder_sample":
                if full_cov:
                    Q_dist = torch.distributions.MultivariateNormal(loc=Emean[:, :, 0,0], scale_tril=E_chol)
                    z0 = Q_dist.sample().unsqueeze(-1)

                else:
                    Q_dist = torch.distributions.Normal(loc=Emean[:, :, 0], scale=torch.sqrt(Evar))
                    z0 = Q_dist.sample()

            elif initial_state=="posterior_mean" or initial_state=="posterior_sample":
                if vae.rnn.params["scalar_noise_z"]=="Cov":
                    E_prec = torch.cholesky_inverse(vae.rnn.chol_cov_embed(log_Evar[:,:,:,0])) #n_trials x dim_z x dim_z 
                    prior_prec_t0 = torch.cholesky_inverse(vae.rnn.chol_cov_embed(vae.rnn.R_z_t0)) #dim_z x dim_z 
                    precQ = prior_prec_t0.unsqueeze(0) + E_prec
                    var_Q = torch.linalg.inv(precQ)
                    var_Q = ((var_Q + torch.permute(var_Q,(0,2,1))) / 2)+torch.eye(vae.dim_z, device=var_Q.device).unsqueeze(0)* 1e-8
                    var_Q_cholesky = torch.linalg.cholesky(var_Q)
            
                    alpha = torch.einsum("Bqz,Bzy->Bqy",var_Q,E_prec)
                    one_min_alpha = torch.eye(vae.dim_z, device=alpha.device).unsqueeze(0) - alpha

                    mean_Q = torch.einsum("Bzs,Bsk->Bzk", one_min_alpha, prior_mean) + torch.einsum(
                        "Bzx,BxK->BzK", alpha, Emean[:, :, 0]
                    )
                    # Sample from posterior and calculate likelihood
                    Q_dist = torch.distributions.MultivariateNormal(
                        loc=mean_Q.permute(0, 2, 1), scale_tril=var_Q_cholesky.unsqueeze(1)
                    )
                    z0 = Q_dist.sample().permute(0, 2, 1)
                else:
                    precZ = 1 / eff_var_prior_t0
                    precE = 1 / Evar
                    #print(precZ.shape)
                    #print(precE.shape)
                    precQ = precZ + precE
                    alpha = precE / precQ
                    mean_Q = (1 - alpha) * prior_mean + alpha * Emean[:, :, 0]
                    eff_var_Q = 1 / precQ
                    Q_dist = torch.distributions.Normal(loc=mean_Q, scale=torch.sqrt(eff_var_Q))
                    z0 = Q_dist.sample()

                if initial_state=="posterior_mean":
                    z0 = mean_Q
     
            else:
                raise ValueError("initial state not recognized")
                
    return z0


def predict(vae,u,x=None,initial_state="prior_sample", max_trials=1000,return_rate=False, return_latent=False,
            normalize_data=False,keep_gradient=False,verbose=True):
    # extract teacher RNN data
    if isinstance(initial_state, str):
        if verbose:
            print("initial state is " + initial_state)
        with torch.no_grad():
            z0 = get_initial_state(vae,u,x,initial_state=initial_state,max_trials=max_trials,normalize_data=normalize_data)
    else:
        z0 = initial_state
    dur = u.shape[2]
    u = u[:max_trials]

    if keep_gradient:
        Z, v = vae.rnn.get_latent_time_series(time_steps=dur, z0=z0, u=u, noise_scale=1,sim_v=True,keep_gradient=True)
        data_gen = (
            vae.rnn.get_observation(Z, v=v,noise_scale=0)
        )
       
        Z = Z[:,:,:,0]
        data_gen_spikes = torch.poisson(data_gen[:,:,:,0])
    else:
        with torch.no_grad():
            Z, v = vae.rnn.get_latent_time_series(time_steps=dur, z0=z0, u=u, noise_scale=1,sim_v=True,keep_gradient=False)
            data_gen = (
                vae.rnn.get_observation(Z, v=v,noise_scale=0)
            )
        Z = Z.cpu().detach().numpy()[:,:,:,0]
        data_gen = data_gen.cpu().detach().numpy()[:, :, :, 0]
        data_gen_spikes = np.random.poisson(data_gen).astype('float64')
    #data_gen_spikes = data_gen

    if return_rate and return_latent:
        return data_gen_spikes, data_gen, Z
    if return_latent:
        return data_gen_spikes, Z
    if return_rate:
        return data_gen_spikes, data_gen 
    return data_gen_spikes

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

