import torch
import numpy as np
from vi_rnn.initialize_parameterize import chol_cov_embed, full_cov_embed
from vi_rnn.inference import Kalman_update_highD, Kalman_update_lowD


def get_initial_state(
    vae, u, x=None, initial_state="prior_sample", optimal_proposal=False, k=1
):
    """
    Sample new data from the model
    Args:
        vae (VAE): trained VAE model
        u (torch.tensor; batch_size x dim_u x dim_T): inputs
        x (torch.tensor; batch_size x dim_x x dim_T): data
        initial_state (str): initial state of the model
        optimal_roposal (bool): whether to use optimal proposal or encoder
        k: number of particles
    Returns:
        z0 (np.array; batch_size x dim_z x k): initial state
    """

    with torch.no_grad():

        # get prior mean and std
        prior_mean = vae.rnn.get_initial_state(u[:, :, 0]).unsqueeze(-1)
        prior_mean = prior_mean.expand(*prior_mean.shape[:2], k)
        if initial_state == "prior_sample":
            if vae.rnn.params["noise_z"] == "full":
                chol_prior_t0 = chol_cov_embed(vae.rnn.R_z_t0)
                Q_dist = torch.distributions.MultivariateNormal(
                    loc=prior_mean.squeeze(-1), scale_tril=chol_prior_t0.unsqueeze(0)
                )
                z0 = Q_dist.sample().unsqueeze(-1)
            else:
                eff_var_prior_t0 = torch.clamp(
                    vae.rnn.var_embed_z_t0(vae.rnn.R_z_t0).unsqueeze(0).unsqueeze(-1),
                    min=vae.min_var,
                    max=vae.max_var,
                )  # 1,Dz,1
                Q_dist = torch.distributions.Normal(
                    loc=prior_mean, scale=torch.sqrt(eff_var_prior_t0)
                )
                z0 = Q_dist.sample()
        elif initial_state == "prior_mean":
            z0 = prior_mean

        elif optimal_proposal:
            eff_var_x = torch.clip(vae.rnn.var_embed_x(vae.rnn.R_x), 1e-8)
            eff_var_prior_t0_chol = chol_cov_embed(vae.rnn.R_z_t0)

            # Get the observation weights and bias
            if vae.rnn.params["readout_from"] == "currents":
                m = vae.rnn.transition.m
                B = vae.rnn.observation.B.unsqueeze(-1) * m
                B = B.T
            else:
                B = vae.rnn.observation.B
            Obs_bias = vae.rnn.observation.Bias.view(1, -1, 1)

            # set Kalman update function
            if vae.dim_x < vae.dim_z * 4:
                kalman_update = Kalman_update_highD
            else:
                kalman_update = Kalman_update_lowD
            alpha, one_min_alpha, Kalman_gain, var_Q_cholesky = kalman_update(
                eff_var_prior_t0_chol, B, eff_var_x
            )

            mean_Q = torch.einsum(
                "zs,BsK->BzK", one_min_alpha, prior_mean
            ) + torch.einsum(
                "zx,BxK->BzK", Kalman_gain, x[:, :, 0].unsqueeze(-1) - Obs_bias
            )

            if initial_state == "posterior_sample":
                Q_dist = torch.distributions.MultivariateNormal(
                    loc=mean_Q.permute(0, 2, 1), scale_tril=var_Q_cholesky
                )
                z0 = Q_dist.sample().permute(0, 2, 1)

            elif initial_state == "posterior_mean":
                z0 = mean_Q

            else:
                raise ValueError(
                    "initial state not recognized, use prior_sample, prior_mean, posterior_sample or posterior_mean"
                )

        else:  # we include the encoder
            x_enc = torch.clone(x)

            Emean, log_Evar = vae.encoder(x_enc)

            Evar = torch.clamp(
                torch.exp(log_Evar[:, :, 0]), min=vae.min_var, max=vae.max_var
            )  # Bs,Dx,T,K
            eff_var_prior_t0 = torch.clamp(
                vae.rnn.var_embed_z_t0(vae.rnn.R_z_t0).unsqueeze(0).unsqueeze(-1),
                min=vae.min_var,
                max=vae.max_var,
            )  # 1,Dz,1
            precZ = 1 / eff_var_prior_t0
            precE = 1 / Evar
            precQ = precZ + precE
            alpha = precE / precQ
            mean_Q = (1 - alpha) * prior_mean + alpha * Emean[:, :, 0]

            if initial_state == "posterior_sample":
                eff_var_Q = 1 / precQ
                Q_dist = torch.distributions.Normal(
                    loc=mean_Q, scale=torch.sqrt(eff_var_Q)
                )
                z0 = Q_dist.sample()

            elif initial_state == "posterior_mean":
                z0 = mean_Q

            else:
                raise ValueError(
                    "initial state not recognized, use prior_sample, prior_mean, posterior_sample or posterior_mean"
                )
    return z0


def generate(
    vae, u=None, x=None, dur=None, initial_state="prior_sample", cut_off=0, k=1
):
    """
    Sample new data from the model
    Args:
        vae (VAE): trained VAE model
        u (torch.tensor; batch_size x dim_u x dim_T): inputs
        x (torch.tensor; batch_size x dim_x x dim_T): data
        dur (int): duration of the simulation
        initial_state (str): initial state of the model
        cut_off (int): cut off for the inputs
        k: number of particles
    Returns:
        Z (np.array; batch_size x dim_z x dim_T): latent variables
        data_gen (np.array; batch_size x dim_x x dim_T): generated data
        rates (np.array; batch_size x dim_x x dim_T): rates underlying generated data
    """
    if vae.has_encoder:
        optimal_proposal = False
    else:
        optimal_proposal = True

    with torch.no_grad():
        if len(x.shape) == 2:
            x = x.unsqueeze(0)  # add trial dim if not used
        if u is None:
            u = torch.zeros(x.shape[0], 0, x.shape[2])
        if dur is None:
            dur = u.shape[2]
        else:
            u = u[:, :, :dur]
        if cut_off > 0:
            u = torch.nn.functional.pad(u, (0, cut_off))
        if isinstance(initial_state, str):
            with torch.no_grad():
                z0 = get_initial_state(
                    vae,
                    u,
                    x,
                    initial_state=initial_state,
                    optimal_proposal=optimal_proposal,
                )
        else:
            z0 = initial_state
        Z, v = vae.rnn.get_latent_time_series(
            time_steps=dur, z0=z0, u=u, noise_scale=1, cut_off=cut_off, k=k
        )

        rates, data_gen = vae.rnn.get_observation(Z, v=v)

    return Z, data_gen, rates
