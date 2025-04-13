import os
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

from encoders import CNN_encoder
from vi_rnn.rnn import RNN
import torch.nn as nn
import torch
import numpy as np
from initialize_parameterize import full_cov_embed, chol_cov_embed


class VAE(nn.Module):
    """
    VAE with low-rank RNN / dynamical systems prior
    """

    def __init__(self, vae_params):
        """initialize"""

        super(VAE, self).__init__()

        # rename some dict keys for backwards compatibility
        # backwards_compat(vae_params)

        self.dim_x = vae_params["dim_x"]
        if "dim_x_hat" in vae_params:
            self.dim_x_hat = vae_params["dim_x_hat"]
        else:
            self.dim_x_hat = vae_params["dim_x"]

        if "dim_u" in vae_params:
            self.dim_u = vae_params["dim_u"]
        else:
            self.dim_u = 0

        self.dim_z = vae_params["dim_z"]
        self.dim_N = vae_params["dim_N"]
        self.vae_params = vae_params
        self.rnn = RNN(
            self.dim_x_hat,
            self.dim_z,
            self.dim_u,
            self.dim_N,
            vae_params["rnn_params"],
        )
        self.has_encoder = True
        if vae_params["enc_architecture"] == "CNN":
            self.encoder = CNN_encoder(self.dim_x, self.dim_z, vae_params["enc_params"])
        print("Loading VAE without encoder")
        self.has_encoder = False

        self.min_var = 1e-6
        self.max_var = 100
        self.MSE_loss = nn.MSELoss()

    def forward_optimal_proposal(self, x, u=None, k=1, resample=False, sim_v=False):
        """
        Forward pass of the VAE
        Note, here the approximate posterior is the optimal linear combination of the encoder and the RNN
        This can be calculated using the Kalman filter for linear observations and non-linear latents
        Args:
            x (torch.tensor; n_trials x dim_X x time_steps): input data
            u (torch.tensor; n_trials x dim_U x time_steps): input stim
            k (int): number of particles
            resample (str): resampling method
            sim_v (bool): simulate the input dynamics
        Returns:
            Loss (torch.tensor; n_trials): loss
            Qzs (torch.tensor; n_trials x dim_z x time_steps): latent time series as predicted by the approximate posterior
            Esample (torch.tensor; n_trials x dim_z x time_steps): latent time series as predicted by the encoder
            log_xs (torch.tensor; n_trials): log likelihood of the data (with averaging over particles in the log)
            log_pzs (torch.tensor; n_trials): log likelihood under the prior (with averaging over particles in the log)
            log_qzs (torch.tensor; n_trials): log likelihood /entropy under the approximate posterior (with averaging over particles in the log)
            log_likelihood (torch.tensor; n_trials): log likelihood (with averaging over particles in the log)
            alphas  (torch.tensor; n_trials x dim_z x dim_z x time_steps): interpolation coefficients
        """
        log_ws = []
        log_ll = []
        ll_xs = []
        ll_pzs = []
        ll_qzs = []
        Qzs = []
        alphas = []

        # project and clamp the variances
        eff_var_prior = full_cov_embed(self.rnn.R_z)
        eff_var_prior_t0 = full_cov_embed(self.rnn.R_z_t0)
        eff_var_prior_chol = chol_cov_embed(self.rnn.R_z)
        eff_var_prior_t0_chol = chol_cov_embed(self.rnn.R_z_t0)

        eff_var_x = torch.clip(self.rnn.var_embed_x(self.rnn.R_x), 1e-8)
        eff_var_x_inv = 1.0 / torch.clip(self.rnn.var_embed_x(self.rnn.R_x), 1e-8)
        eff_std_x = torch.clip(self.rnn.std_embed_x(self.rnn.R_x), 1e-4)

        batch_size, dim_x, time_steps = x.shape
        dim_z = self.dim_z

        # Get the initial prior mean
        if sim_v:
            prior_mean = (
                self.rnn.get_initial_state(torch.zeros_like(u[:, :, 0])).unsqueeze(2)
                .expand(batch_size, self.dim_z, k)
            )
            v = torch.zeros(batch_size, self.dim_u, 1, device=x.device)

        else:  # initialise in the affine subspace corresponding to the input
            prior_mean = (
                self.rnn.get_initial_state(u[:, :, 0]).unsqueeze(2)
                .expand(batch_size, self.dim_z, k)
            )  # BS,Dz,K
            v = u[:, :, 0].unsqueeze(-1)  # add particle dimension

        x = x.unsqueeze(-1)  # add particle dimension

        # Get the observation weights and bias
        if self.rnn.params["readout_from"] == "currents":
            m = self.rnn.transition.m
            B = self.rnn.observation.cast_B(self.rnn.observation.B).T @ m
            B = B.T
        else:
            B = self.rnn.observation.cast_B(self.rnn.observation.B)
        Obs_bias = self.rnn.observation.Bias.squeeze(-1)

        # Calculate the Kalman gain and interpolation alpha
        if dim_x < dim_z * 4:
            Kalman_gain = (
                eff_var_prior_t0
                @ B
                @ torch.linalg.inv(torch.diag(eff_var_x) + B.T @ eff_var_prior_t0 @ B)
            )
            alpha = Kalman_gain @ B.T
            one_min_alpha = torch.eye(self.dim_z, device=alpha.device) - alpha

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
            one_min_alpha = torch.eye(self.dim_z, device=alpha.device) - alpha

        # avoid numerical issues
        var_Q = (
            torch.eye(self.dim_z, device=alpha.device) * 1e-8 + (var_Q + var_Q.T) / 2
        )

        var_Q_cholesky = torch.linalg.cholesky(var_Q)
        # Posterior Mean

        mean_Q = torch.einsum("zs,BsK->BzK", one_min_alpha, prior_mean) + torch.einsum(
            "zx,BxK->BzK", Kalman_gain, x[:, :, 0] - Obs_bias
        )

        # Sample from posterior and calculate likelihood
        Q_dist = torch.distributions.MultivariateNormal(
            loc=mean_Q.permute(0, 2, 1), scale_tril=var_Q_cholesky
        )
        Qz = Q_dist.rsample()

        ll_qz = Q_dist.log_prob(Qz)

        # Calculate likelihood under the prior
        pz_dist = torch.distributions.MultivariateNormal(
            loc=prior_mean.permute(0, 2, 1), scale_tril=eff_var_prior_t0_chol
        )
        ll_pz = pz_dist.log_prob(Qz)

        # Get observation mean and calculate likelihood of the data
        Qz = Qz.permute(0, 2, 1)
        mean_x = torch.einsum("zx, bzk -> bxk", B, Qz) + Obs_bias
        x_dist = torch.distributions.Normal(
            loc=mean_x.permute(0, 2, 1), scale=eff_std_x
        )
        ll_x = x_dist.log_prob(x[:, :, 0].permute(0, 2, 1)).sum(axis=-1)

        # Calculate the log weights
        log_w = ll_x + ll_pz - ll_qz
        # Store some quantities
        ll_xsum = torch.logsumexp(ll_x.detach(), axis=-1) - np.log(k)
        ll_pzsum = torch.logsumexp(ll_pz.detach(), axis=-1) - np.log(k)
        ll_qzsum = torch.logsumexp(ll_qz.detach(), axis=-1) - np.log(k)
        log_ws.append(torch.logsumexp(log_w, axis=-1) - np.log(k))
        log_ll.append(torch.logsumexp(log_w.detach(), axis=-1) - np.log(k))
        Qzs.append(Qz)
        ll_xs.append(ll_xsum)
        ll_pzs.append(ll_pzsum)
        ll_qzs.append(ll_qzsum)

        time_steps = x.shape[2]
        u = u.unsqueeze(-1)  # account for k

        # precalculate Kalman Gain / Interpolation
        if dim_x < dim_z * 4:
            Kalman_gain = (
                eff_var_prior
                @ B
                @ torch.linalg.inv(torch.diag(eff_var_x) + B.T @ eff_var_prior @ B)
            )
            alpha = Kalman_gain @ B.T
            one_min_alpha = torch.eye(self.dim_z, device=alpha.device) - alpha
            # Posterior Joseph stabilised Covariance
            var_Q = (
                one_min_alpha @ eff_var_prior @ one_min_alpha.T
                + (Kalman_gain * torch.unsqueeze(eff_var_x, 0)) @ Kalman_gain.T
            )
        else:
            var_Q = torch.linalg.inv(
                torch.cholesky_inverse(eff_var_prior_chol)
                + (B * torch.unsqueeze(eff_var_x_inv, 0)) @ B.T
            )
            Kalman_gain = var_Q @ (B * torch.unsqueeze(eff_var_x_inv, 0))
            alpha = Kalman_gain @ B.T
            one_min_alpha = torch.eye(self.dim_z, device=alpha.device) - alpha

        var_Q = (
            torch.eye(self.dim_z, device=alpha.device) * 1e-8 + (var_Q + var_Q.T) / 2
        )
        var_Q_cholesky = torch.linalg.cholesky(var_Q)

        # Start the loop through the time steps
        for t in range(1, time_steps):
            # Resample if necessary
            if resample == "multinomial":
                indices = sample_indices_multinomial(log_w)
                Qz = resample_Q(Qz, indices)
            elif resample == "systematic":
                indices = sample_indices_systematic(log_w)
                Qz = resample_Q(Qz, indices)
            elif resample == "none":
                pass
            else:
                print("WARNING: resample does not exist")
                print("use, one of: multinomial, systematic, none")

            # Get the prior mean
            prior_mean = self.rnn.transition(Qz, v=v)
            # progress input dynamics
            if sim_v:
                v = self.rnn.transition.step_input(v, u[:, :, t - 1])
            else:
                v = u[:, :, t]
            # Calculate the Kalman gain and interpolation alpha

            # Calculate the posterior mean and Joseph stabilised covariance
            if self.rnn.params["readout_from"] == "currents":
                v_to_X = torch.einsum(
                    "xv, bvk -> bxk", self.rnn.transition.Wu, v.squeeze(-2)
                )
            else:
                v_to_X = 0

            x_t = x[:, :, t] - Obs_bias - v_to_X
            mean_Q = torch.einsum(
                "zs,BsK->BzK", one_min_alpha, prior_mean
            ) + torch.einsum("zx,BxK->BzK", Kalman_gain, x_t)

            # Sample from the posterior and calculate likelihood
            Q_dist = torch.distributions.MultivariateNormal(
                loc=mean_Q.permute(0, 2, 1), scale_tril=var_Q_cholesky
            )
            Qz = Q_dist.rsample()
            ll_qz = Q_dist.log_prob(Qz)
            
            # Calculate likelihood under the prior
            pz_dist = torch.distributions.MultivariateNormal(
                loc=prior_mean.permute(0, 2, 1), scale_tril=eff_var_prior_chol
            )
            ll_pz = pz_dist.log_prob(Qz)

            # Get observation mean and calculate likelihood of the data
            Qz = Qz.permute(0, 2, 1)
            mean_x = torch.einsum("zx, bzk -> bxk", B, Qz) + Obs_bias + v_to_X

            x_dist = torch.distributions.Normal(
                loc=mean_x.permute(0, 2, 1), scale=eff_std_x
            )
            ll_x = x_dist.log_prob(x[:, :, t].permute(0, 2, 1)).sum(axis=-1)

            # Calculate the log weights
            log_w = ll_x + ll_pz - ll_qz

            # weights also have analytic expression: https://www.ecmwf.int/sites/default/files/elibrary/2012/76468-particle-filters-optimal-proposal-and-high-dimensional-systems_0.pdf
            # w_mean = torch.einsum("zx, bzk -> bxk", B, prior_mean) + Obs_bias + v_to_X
            # w_upd =  torch.einsum("zx, zs, sy -> xy", B, eff_var_prior, B)+torch.diag(eff_var_x)
            # w_chol = torch.linalg.cholesky(w_upd)
            # w_dist = torch.distributions.MultivariateNormal(loc=w_mean.permute(0, 2, 1), scale_tril=w_chol#)
            # ll_w = w_dist.log_prob(x[:, :, t].permute(0, 2, 1))
        
            # Store some quantities
            ll_xsum = torch.logsumexp(ll_x.detach(), axis=-1) - np.log(k)
            ll_pzsum = torch.logsumexp(ll_pz.detach(), axis=-1) - np.log(k)
            ll_qzsum = torch.logsumexp(ll_qz.detach(), axis=-1) - np.log(k)
            log_ws.append(torch.logsumexp(log_w, axis=-1) - np.log(k))
            log_ll.append(torch.logsumexp(log_w.detach(), axis=-1) - np.log(k))
            Qzs.append(Qz)
            ll_xs.append(ll_xsum)
            ll_pzs.append(ll_pzsum)
            ll_qzs.append(ll_qzsum)
            alphas.append(alpha)

        # Make tensors from lists
        log_ws = torch.stack(log_ws)
        log_ll = torch.stack(log_ll)
        log_xs = torch.stack(ll_xs)
        log_pzs = torch.stack(ll_pzs)
        log_qzs = torch.stack(ll_qzs)
        alphas = torch.stack(alphas)

        # Average over time steps
        log_likelihood = torch.sum(log_ll, axis=0)
        Loss = torch.sum(log_ws, axis=0)
        log_xs = torch.sum(log_xs, axis=0)
        log_pzs = torch.sum(log_pzs, axis=0)
        log_qzs = torch.sum(log_qzs, axis=0)
        log_likelihood /= time_steps
        Loss /= time_steps
        log_xs /= time_steps
        log_pzs /= time_steps
        log_qzs /= time_steps

        Qzs = torch.stack(Qzs)
        Qzs = Qzs.permute(1, 2, 0, 3)

        return (
            Loss,
            Qzs,
            log_xs,
            log_pzs,
            -log_qzs,
            log_likelihood,
            alphas,
        )

    def forward(
        self,
        x,
        u=None,
        k=1,
        resample=False,
        out_likelihood="Gauss",
        t_forward=0,
        sim_v=False,
    ):
        """
        Forward pass of the VAE
        Note, here the approximate posterior is a linear combination of the encoder and the RNN
        Args:
            x (torch.tensor; n_trials x dim_X x time_steps): input data
            u (torch.tensor; n_trials x dim_U x time_steps): input stim
            k (int): number of particles
            resample (str): resampling method
            out_likelihood (str): likelihood of the output
            t_forward (int): number of time steps to predict forward without using the encoder
            sim_v (bool): simulate the input dynamics
        Returns:
            Loss (torch.tensor; n_trials): loss
            Qzs (torch.tensor; n_trials x dim_z x time_steps): latent time series as predicted by the approximate posterior
            Esample (torch.tensor; n_trials x dim_z x time_steps): latent time series as predicted by the encoder
            log_xs (torch.tensor; n_trials): log likelihood of the data (with averaging over particles in the log)
            log_pzs (torch.tensor; n_trials): log likelihood under the prior (with averaging over particles in the log)
            log_qzs (torch.tensor; n_trials): log likelihood /entropy under the approximate posterior (with averaging over particles in the log)
            log_likelihood (torch.tensor; n_trials): log likelihood (with averaging over particles in the log)
            alphas (torch.tensor; n_trials x dim_z x time_steps): interpolation coefficients

        """
        batch_size = x.shape[0]
        # Define the data likelihood function
        if out_likelihood == "Gauss":
            ll_x_func = (
                lambda x, mu, sd: torch.distributions.Normal(loc=mu, scale=sd)
                .log_prob(x)
                .sum(axis=1)
            )
        elif out_likelihood == "Poisson":
            ll_x_func = (
                lambda x, mu, sd: torch.distributions.Poisson(mu)
                .log_prob(x)
                .sum(axis=1)
            )
        else:
            print("WARNING: likelihood does not exist, use one of: Gauss, Poisson")

        # Run the encoder
        Emean, log_Evar = self.encoder(
            x[:, : self.dim_x, : x.shape[2] - t_forward], k=k
        )  # Bs,Dx,T,K

        # Project and clamp the variances
        Evar = torch.clamp(torch.exp(log_Evar), min=self.min_var, max=self.max_var)
        eff_var_prior = torch.clamp(
            self.rnn.var_embed_z(self.rnn.R_z).unsqueeze(0).unsqueeze(-1),
            min=self.min_var,
            max=self.max_var,
        )  # 1,Dz,1
        eff_std_prior = torch.clamp(
            self.rnn.std_embed_z(self.rnn.R_z).unsqueeze(0).unsqueeze(-1),
            min=np.sqrt(self.min_var),
            max=np.sqrt(self.max_var),
        )  # 1,Dz,1
        eff_var_prior_t0 = torch.clamp(
            self.rnn.var_embed_z_t0(self.rnn.R_z_t0).unsqueeze(0).unsqueeze(-1),
            min=self.min_var,
            max=self.max_var,
        )  # 1,Dz,1
        eff_std_prior_t0 = torch.clamp(
            self.rnn.std_embed_z_t0(self.rnn.R_z_t0).unsqueeze(0).unsqueeze(-1),
            min=np.sqrt(self.min_var),
            max=np.sqrt(self.max_var),
        )  # 1,Dz,1
        eff_std_x = torch.clamp(
            self.rnn.std_embed_x(self.rnn.R_x).unsqueeze(0).unsqueeze(-1),
            min=np.sqrt(self.min_var),
            max=np.sqrt(self.max_var),
        )  # 1,Dx,1

        # Cut some of the data if a CNN was used without padding

        x_hat = x.unsqueeze(-1)

        # Initialise some lists
        bs, dim_z, time_steps, _ = Emean.shape
        log_ws = []
        log_ll = []
        ll_xs = []
        ll_pzs = []
        ll_qzs = []
        Qzs = []
        alphas = []

        # Get the initial prior mean
        prior_mean = (
            self.rnn.get_initial_state(u[:, :, 0])
            .unsqueeze(2)
            .expand(batch_size, self.dim_z, k)
        )  # BS,Dz,K

        if sim_v:
            v = torch.zeros(batch_size, self.dim_u, 1, 1, device=x.device)
        else:
            v = u[:, :, 0].unsqueeze(-1).unsqueeze(-1)
        # Calculate the initial posterior mean and covariance

        precZ = 1 / eff_var_prior_t0
        precE = 1 / Evar[:, :, 0]
        precQ = precZ + precE
        alpha = precE / precQ
        mean_Q = (1 - alpha) * prior_mean + alpha * Emean[:, :, 0]
        eff_var_Q = 1 / precQ

        # Sample from the posterior and calculate likelihood
        Q_dist = torch.distributions.Normal(loc=mean_Q, scale=torch.sqrt(eff_var_Q))
        Qz = Q_dist.rsample()
        ll_qz = Q_dist.log_prob(Qz).sum(axis=1)

        # Calculate the log likelihood under the prior
        ll_pz = (
            torch.distributions.Normal(loc=prior_mean, scale=eff_std_prior_t0)
            .log_prob(Qz)
            .sum(axis=1)
        )

        # Get the observation mean and calculate likelihood of the data
        mean_x = self.rnn.get_observation(Qz, noise_scale=0, v=v)

        ll_x = ll_x_func(x_hat[:, :, 0], mean_x, eff_std_x)

        # Calculate the log weights
        log_w = ll_x + ll_pz - ll_qz

        # Store some quantities
        log_ll.append(torch.logsumexp(log_w.detach(), axis=-1) - np.log(k))
        ll_xs.append(torch.logsumexp(ll_x.detach(), axis=-1) - np.log(k))
        ll_pzs.append(torch.logsumexp(ll_pz.detach(), axis=-1) - np.log(k))
        ll_qzs.append(torch.logsumexp(ll_qz.detach(), axis=-1) - np.log(k))
        alphas.append(alpha)
        log_ws.append(torch.logsumexp(log_w, axis=-1) - np.log(k))
        Qzs.append(Qz)

        u = u.unsqueeze(-1)  # add particle dimension

        # Loop through the time steps
        for t in range(1, time_steps):

            # Resample if necessary
            if resample == "multinomial":
                indices = sample_indices_multinomial(log_w)
                Qz = resample_Q(Qz, indices)
            elif resample == "systematic":
                indices = sample_indices_systematic(log_w)
                Qz = resample_Q(Qz, indices)
            elif resample == "none":
                pass
            else:
                print("WARNING: resample does not exist")
                print("use, one of: multinomial, systematic, none")

            # Get the prior mean
            prior_mean = self.rnn.transition(Qz, v=v)
            if sim_v:
                v = self.rnn.transition.step_input(v, u[:, :, t - 1])
            else:
                v = u[:, :, t]

            # Calculate the posterior mean and covariance
            precZ = 1 / eff_var_prior
            precE = 1 / Evar[:, :, t]
            precQ = precZ + precE
            alpha = precE / precQ
            alphas.append(alpha)
            mean_Q = (1 - alpha) * prior_mean + alpha * Emean[:, :, t]
            eff_var_Q = 1 / precQ

            # Sample from the posterior and calculate likelihood
            Q_dist = torch.distributions.Normal(loc=mean_Q, scale=torch.sqrt(eff_var_Q))
            Qz = Q_dist.rsample()
            ll_qz = Q_dist.log_prob(Qz).sum(axis=1)

            # Calculate the log likelihood under the prior
            ll_pz = (
                torch.distributions.Normal(loc=prior_mean, scale=eff_std_prior)
                .log_prob(Qz)
                .sum(axis=1)
            )

            # Get the observation mean and calculate likelihood of the data
            mean_x = self.rnn.get_observation(
                Qz.unsqueeze(-2), noise_scale=0, v=v
            ).squeeze(-2)
            ll_x = ll_x_func(x_hat[:, :, t], mean_x, eff_std_x)

            # Calculate the log weights
            log_w = ll_x + ll_pz - ll_qz

            # Store some quantities
            log_ll.append(torch.logsumexp(log_w.detach(), axis=-1) - np.log(k))
            ll_xs.append(torch.logsumexp(ll_x.detach(), axis=-1) - np.log(k))
            ll_pzs.append(torch.logsumexp(ll_pz.detach(), axis=-1) - np.log(k))
            ll_qzs.append(torch.logsumexp(ll_qz.detach(), axis=-1) - np.log(k))
            log_ws.append(torch.logsumexp(log_w, axis=-1) - np.log(k))
            Qzs.append(Qz)

        # Use Bootstrap samples for the last t_forward steps
        for t in range(time_steps, time_steps + t_forward):
            if resample == "multinomial":
                indices = sample_indices_multinomial(log_w)
                Qz = resample_Q(Qz, indices)
            elif resample == "systematic":
                indices = sample_indices_systematic(log_w)
                Qz = resample_Q(Qz, indices)

            elif resample == "none":
                pass
            else:
                print("WARNING: resample does not exist")
                print("use, one of: multinomial, systematic, none")

            # Here prior and posterior are the same and we just need the likelihood of the data
            prior_mean = self.rnn.transition(Qz, v=v).squeeze(2)
            if sim_v:
                v = self.rnn.transition.step_input(v, u[:, :, t - 1])
            else:
                v = u[:, :, t]

            # Sample from the posterior and calculate likelihood
            Q_dist = torch.distributions.Normal(
                loc=prior_mean, scale=torch.sqrt(eff_var_prior)
            )
            Qz = Q_dist.rsample()

            mean_x = self.rnn.get_observation(Qz.unsqueeze(-2), noise_scale=0).squeeze(
                -2
            )

            ll_pz = (
                torch.distributions.Normal(loc=prior_mean, scale=eff_std_prior)
                .log_prob(Qz)
                .sum(axis=1)
            )

            ll_x = ll_x_func(x_hat[:, :, t], mean_x, eff_std_x)
            log_w = ll_x
            ll_qz = ll_qz

            # Store some quantities
            log_ll.append(torch.logsumexp(log_w.detach(), axis=-1) - np.log(k))
            ll_xs.append(torch.logsumexp(ll_x.detach(), axis=-1) - np.log(k))
            ll_pzs.append(torch.logsumexp(ll_pz.detach(), axis=-1) - np.log(k))
            ll_qzs.append(torch.logsumexp(ll_qz.detach(), axis=-1) - np.log(k))
            log_ws.append(torch.logsumexp(log_w, axis=-1) - np.log(k))
            Qzs.append(Qz)

        # Make tensors from lists
        log_ws = torch.stack(log_ws)
        log_ll = torch.stack(log_ll)
        log_xs = torch.stack(ll_xs)
        log_pzs = torch.stack(ll_pzs)
        log_qzs = torch.stack(ll_qzs)
        alphas = torch.stack(alphas)

        # Average over time steps
        log_likelihood = torch.mean(log_ll, axis=0)
        Loss = torch.mean(log_ws, axis=0)
        log_xs = torch.mean(log_xs, axis=0)
        log_pzs = torch.mean(log_pzs, axis=0)
        log_qzs = torch.mean(log_qzs, axis=0)

        Qzs = torch.stack(Qzs)
        Qzs = Qzs.permute(1, 2, 0, 3)

        return Loss, Qzs, log_xs, log_pzs, -log_qzs, log_likelihood, alphas

    def forward_bootstrap_proposal(
        self,
        x,
        u=None,
        k=1,
        resample=False,
        out_likelihood="Gauss",
        t_forward=0,
        sim_v=False,
    ):
        """
        Forward pass of the VAE
        Note, here the approximate posterior is just the RNN
        Args:
            x (torch.tensor; n_trials x dim_X x time_steps): input data
            u (torch.tensor; n_trials x dim_U x time_steps): input stim
            k (int): number of particles
            resample (str): resampling method
            out_likelihood (str): likelihood of the output
            t_forward (int): number of time steps to predict forward without using the encoder
            sim_v (bool): simulate the input dynamics
        Returns:
            Loss (torch.tensor; n_trials): loss
            Qzs (torch.tensor; n_trials x dim_z x time_steps): latent time series as predicted by the approximate posterior
            Esample (torch.tensor; n_trials x dim_z x time_steps): latent time series as predicted by the encoder
            log_xs (torch.tensor; n_trials): log likelihood of the data (with averaging over particles in the log)
            log_pzs (torch.tensor; n_trials): log likelihood under the prior (with averaging over particles in the log)
            log_qzs (torch.tensor; n_trials): log likelihood /entropy under the approximate posterior (with averaging over particles in the log)
            log_likelihood (torch.tensor; n_trials): log likelihood (with averaging over particles in the log)
            alphas (torch.tensor; n_trials x dim_z x time_steps): interpolation coefficients

        """

        # Define the data likelihood function
        if out_likelihood == "Gauss":
            ll_x_func = (
                lambda x, mu, sd: torch.distributions.Normal(loc=mu, scale=sd)
                .log_prob(x)
                .sum(axis=1)
            )
        elif out_likelihood == "Poisson":
            ll_x_func = (
                lambda x, mu, sd: torch.distributions.Poisson(mu)
                .log_prob(x)
                .sum(axis=1)
            )
        else:
            print("WARNING: likelihood does not exist, use one of: Gauss, Poisson")

        # Project and clamp the variances
        eff_std_prior = torch.clamp(
            self.rnn.std_embed_z(self.rnn.R_z).unsqueeze(0).unsqueeze(-1),
            min=np.sqrt(self.min_var),
            max=np.sqrt(self.max_var),
        )  # 1,Dz,1
        eff_std_prior_t0 = torch.clamp(
            self.rnn.std_embed_z_t0(self.rnn.R_z_t0).unsqueeze(0).unsqueeze(-1),
            min=np.sqrt(self.min_var),
            max=np.sqrt(self.max_var),
        )  # 1,Dz,1
        eff_std_x = torch.clamp(
            self.rnn.std_embed_x(self.rnn.R_x).unsqueeze(0).unsqueeze(-1),
            min=np.sqrt(self.min_var),
            max=np.sqrt(self.max_var),
        )  # 1,Dx,1

        x_hat = x.unsqueeze(-1)

        # Initialise some lists
        log_ws = []
        log_ll = []
        ll_xs = []
        Qzs = []

        # Get the initial prior mean
        batch_size, dim_x, time_steps = x.shape

        prior_mean = (
            self.rnn.get_initial_state(u[:, :, 0])
            .unsqueeze(2)
            .expand(batch_size, self.dim_z, k)
        )  # BS,Dz,K

        if sim_v:
            v = torch.zeros(batch_size, self.dim_u, 1, 1, device=x.device)
        else:
            v = u[:, :, 0].unsqueeze(-1).unsqueeze(-1)
        # Calculate the initial posterior mean and covariance

        # Sample from the posterior and calculate likelihood
        Q_dist = torch.distributions.Normal(loc=prior_mean, scale=eff_std_prior_t0)
        Qz = Q_dist.rsample()

        # Get the observation mean and calculate likelihood of the data
        mean_x = self.rnn.get_observation(Qz.unsqueeze(-2), noise_scale=0).squeeze(-2)
        ll_x = ll_x_func(x_hat[:, :, 0], mean_x, eff_std_x)

        # Calculate the log weights
        log_w = ll_x

        # Store some quantities
        log_ll.append(torch.logsumexp(log_w.detach(), axis=-1) - np.log(k))
        ll_xs.append(torch.logsumexp(ll_x.detach(), axis=-1) - np.log(k))
        log_ws.append(torch.logsumexp(log_w, axis=-1) - np.log(k))
        Qzs.append(Qz)

        u = u.unsqueeze(-1)  # add particle dimension

        # Loop through the time steps
        for t in range(1, time_steps + t_forward):
            # Resample if necessary
            if resample == "multinomial":
                indices = sample_indices_multinomial(log_w)
                Qz = resample_Q(Qz, indices)
            elif resample == "systematic":
                indices = sample_indices_systematic(log_w)
                Qz = resample_Q(Qz, indices)
            elif resample == "none":
                pass
            else:
                print("WARNING: resample does not exist")
                print("use, one of: multinomial, systematic, none")

            # Get the prior mean
            prior_mean = self.rnn.transition(Qz, v=v)
            if sim_v:
                v = self.rnn.transition.step_input(v, u[:, :, t - 1])
            else:
                v = u[:, :, t]

            # Sample from the posterior and calculate likelihood
            Q_dist = torch.distributions.Normal(loc=prior_mean, scale=eff_std_prior)
            Qz = Q_dist.rsample()

            # Get the observation mean and calculate likelihood of the data
            mean_x = self.rnn.get_observation(Qz.unsqueeze(-2), noise_scale=0).squeeze(
                -2
            )
            ll_x = ll_x_func(x_hat[:, :, t], mean_x, eff_std_x)

            # Calculate the log weights
            log_w = ll_x

            # Store some quantities
            log_ll.append(torch.logsumexp(log_w.detach(), axis=-1) - np.log(k))
            ll_xs.append(torch.logsumexp(ll_x.detach(), axis=-1) - np.log(k))
            log_ws.append(torch.logsumexp(log_w, axis=-1) - np.log(k))
            Qzs.append(Qz)

        # Make tensors from lists
        log_ws = torch.stack(log_ws)
        log_ll = torch.stack(log_ll)
        log_xs = torch.stack(ll_xs)

        # Average over time steps
        log_likelihood = torch.mean(log_ll, axis=0)
        Loss = torch.mean(log_ws, axis=0)
        log_xs = torch.mean(log_xs, axis=0)

        Qzs = torch.stack(Qzs)
        Qzs = Qzs.permute(1, 2, 0, 3)
        empty = torch.ones(0, device=Qzs.device)
        return Loss, Qzs, log_xs, empty, empty, log_likelihood, empty

    def to_device(self, device):
        """Move network between cpu / gpu (cuda)"""
        self.rnn.to(device=device)
        self.rnn.normal.loc = self.rnn.normal.loc.to(device=device)
        self.rnn.normal.scale = self.rnn.normal.scale.to(device=device)
        self.rnn.observation.mask = self.rnn.observation.mask.to(device=device)
        self.rnn.transition.Wu = self.rnn.transition.Wu.to(device=device)
        if self.has_encoder:
            self.encoder.to(device=device)

    def posterior(
        self,
        x,
        u=None,
        k=1,
        t_held_in=0,
        t_forward=0,
        resample="systematic",
        marginal_smoothing=False,
        sim_v=False,
    ):
        return self.predict_NLB(
            x, u, k, t_held_in, t_forward, resample, marginal_smoothing, sim_v=sim_v
        )

    def predict_NLB(
        self,
        x,
        u=None,
        k=1,
        t_held_in=None,
        t_forward=0,
        resample="systematic",
        marginal_smoothing=False,
        sim_v=False,
    ):
        """
        Obtain filtering and smoothing posteriors given data and optionally input

        Args:
            x (torch.tensor; n_trials x dim_X x time_steps): input data
            u (torch.tensor; n_trials x dim_U x time_steps): input stim
            k (int): number of particles
            t_held_in (int): number of time steps where the data / encoder is used
            t_forward (int): number of time steps to predict forward without using the encoder
            resample (str): resampling method
            marginal_smoothing (bool): whether or not to obtain the marginal latents

        Returns:
            Qzs_filt (torch.tensor; n_trials x dim_z x time_steps x k): filtered latent time series
            Qzs_smooth (torch.tensor; n_trials x dim_z x time_steps x k): smoothed latent time series
            Xs_filt (torch.tensor; n_trials x dim_x x time_steps x k): filtered observation time series
            Xs_smooth (torch.tensor; n_trials x dim_x x time_steps x k): smoothed observation time series
        """

        if t_held_in is None:
            t_held_in = x.shape[2]

        # no need for gradients here
        self.eval()

        # define the data likelihood function
        ll_x_func = (
            lambda x, mu: torch.distributions.Poisson(mu).log_prob(x).sum(axis=1)
        )

        # Run the encoder
        Emean, log_Evar = self.encoder(x[:, : self.dim_x, :t_held_in], k=k)  # Bs,Dx,T,K
        x_hat = x.unsqueeze(-1)

        # Project and clamp the variances
        Evar = torch.clamp(torch.exp(log_Evar), min=self.min_var, max=self.max_var)

        eff_var_prior = torch.clamp(
            self.rnn.var_embed_z(self.rnn.R_z).unsqueeze(0).unsqueeze(-1),
            min=self.min_var,
            max=self.max_var,
        )  # 1,Dz,1
        eff_std_prior = torch.clamp(
            self.rnn.std_embed_z(self.rnn.R_z).unsqueeze(0).unsqueeze(-1),
            min=np.sqrt(self.min_var),
            max=np.sqrt(self.max_var),
        )  # 1,Dz,1
        eff_var_prior_t0 = torch.clamp(
            self.rnn.var_embed_z_t0(self.rnn.R_z_t0).unsqueeze(0).unsqueeze(-1),
            min=self.min_var,
            max=self.max_var,
        )  # 1,Dz,1
        eff_std_prior_t0 = torch.clamp(
            self.rnn.std_embed_z_t0(self.rnn.R_z_t0).unsqueeze(0).unsqueeze(-1),
            min=np.sqrt(self.min_var),
            max=np.sqrt(self.max_var),
        )  # 1,Dz,1
        bs, dim_z, time_steps, _ = Emean.shape

        # Initialise some lists
        log_ws = []
        Qzs = []
        Qzs_filt = []

        # Get the prior mean and observation mean
        if u is None:
            u = torch.zeros(x.shape[0], self.dim_u, x.shape[2]).to(x.device)
        # Get the initial prior mean
        if sim_v:
            prior_mean = (
                self.rnn.get_initial_state(torch.zeros_like(u[:, :, 0])).unsqueeze(2)
                .expand(bs, self.dim_z, k)
            )
            v = torch.zeros(bs, self.dim_u, 1, device=x.device)
        else:  # initialise in the affine subspace corresponding to the input
            prior_mean = (
                self.rnn.get_initial_state(u[:, :, 0]).unsqueeze(2)
                .expand(bs, self.dim_z, k)
            )  # BS,Dz,K
            v = u[:, :, 0].unsqueeze(-1)  # add particle dimension

        # get the posterior mean and covariance
        precZ = 1 / eff_var_prior_t0
        precE = 1 / Evar[:, :, 0]
        precQ = precZ + precE
        alpha = precE / precQ
        mean_Q = (1 - alpha) * prior_mean + alpha * Emean[:, :, 0]
        eff_var_Q = 1 / precQ

        # Sample from the posterior and calculate likelihood
        Q_dist = torch.distributions.Normal(loc=mean_Q, scale=torch.sqrt(eff_var_Q))
        Qz = Q_dist.rsample()
        ll_qz = Q_dist.log_prob(Qz).sum(axis=1)

        # Calculate the log likelihood under the prior
        ll_pz = (
            torch.distributions.Normal(loc=prior_mean, scale=eff_std_prior_t0)
            .log_prob(Qz)
            .sum(axis=1)
        )

        # Get the observation mean and calculate likelihood of the data
        mean_x = self.rnn.get_observation(Qz.unsqueeze(-2), noise_scale=0, v=v).squeeze(
            -2
        )
        ll_x = ll_x_func(x_hat[:, :, 0], mean_x[:, : x_hat.shape[1]])

        # Calculate the log weights
        log_w = ll_x + ll_pz - ll_qz
        vs = [v.squeeze(2)]

        # Append the log weights and log likelihoods
        log_ws.append(log_w)
        Qzs.append(Qz)
        t_bs = 0
        u = u.unsqueeze(-1)  # add particle dimension

        # Loop through the time steps
        for t in range(1, t_held_in):

            # Resample if necessary
            if resample == "multinomial":
                indices = sample_indices_multinomial(log_w)
                Qz = resample_Q(Qz, indices)
                Qzs_filt.append(Qz)
            elif resample == "systematic":
                indices = sample_indices_systematic(log_w)
                Qz = resample_Q(Qz, indices)
                Qzs_filt.append(Qz)
            elif resample == "none":
                pass
            else:
                print("WARNING: resample does not exist")
                print("use, one of: multinomial, systematic, none")

            # Get the prior mean
            prior_mean = self.rnn.transition(Qz, v=v)
            prior_mean = prior_mean
            if sim_v:
                v = self.rnn.transition.step_input(v, u[:, :, t - 1])
            else:
                v = u[:, :, t]
            vs.append(v)

            # Calculate the posterior mean and covariance
            precZ = 1 / eff_var_prior
            precE = 1 / Evar[:, :, t]
            precQ = precZ + precE
            alpha = precE / precQ
            mean_Q = (1 - alpha) * prior_mean + alpha * Emean[:, :, t]
            eff_var_Q = 1 / precQ

            # Sample from the posterior and calculate likelihood
            Q_dist = torch.distributions.Normal(loc=mean_Q, scale=torch.sqrt(eff_var_Q))
            Qz = Q_dist.rsample()
            ll_qz = Q_dist.log_prob(Qz).sum(axis=1)

            # Calculate the log likelihood under the prior
            ll_pz = (
                torch.distributions.Normal(loc=prior_mean, scale=eff_std_prior)
                .log_prob(Qz)
                .sum(axis=1)
            )

            # Get the observation mean and calculate likelihood of the data
            mean_x = self.rnn.get_observation(
                Qz.unsqueeze(-2), noise_scale=0, v=v
            ).squeeze(-2)
            ll_x = ll_x_func(x_hat[:, :, t], mean_x[:, : x_hat.shape[1]])

            # Calculate the log weights
            log_w = ll_x + ll_pz - ll_qz
            log_ws.append(log_w)
            Qzs.append(Qz)
        for t in range(t_held_in, t_held_in + t_bs):
            # print("bs")
            if resample == "multinomial":
                indices = sample_indices_multinomial(log_w)
                Qz = resample_Q(Qz, indices)
                Qzs_filt.append(Qz)

            elif resample == "systematic":
                indices = sample_indices_systematic(log_w)
                Qz = resample_Q(Qz, indices)
                Qzs_filt.append(Qz)

            elif resample == "none":
                pass
            else:
                print("WARNING: resample does not exist")
                print("use, one of: multinomial, systematic, none")

            # Here prior and posterior are the same and we just need the likelihood of the data
            prior_mean = self.rnn.transition(Qz, v=v)
            if sim_v:
                v = self.rnn.transition.step_input(v, torch.zeros_like(v))
            vs.append(v)
            # Sample from the posterior and calculate likelihood
            Q_dist = torch.distributions.Normal(
                loc=prior_mean, scale=torch.sqrt(eff_var_prior)
            )
            Qz = Q_dist.rsample()

            mean_x = self.rnn.get_observation(
                Qz.unsqueeze(-2), noise_scale=0, v=v
            ).squeeze(-2)

            ll_x = ll_x_func(x_hat[:, :, t], mean_x[:, : x_hat.shape[1]])
            log_w = ll_x
            ll_qz = ll_qz

            # Store some quantities
            Qzs.append(Qz)

        # resample last time steps
        if resample == "multinomial":
            indices = sample_indices_multinomial(log_w)
            Qz = resample_Q(Qz, indices)
            Qzs_filt.append(Qz)

        elif resample == "systematic":
            indices = sample_indices_systematic(log_w)
            Qz = resample_Q(Qz, indices)
            Qzs_filt.append(Qz)

        # Backward Smoothing
        Qzs_sm = torch.zeros(
            t_held_in + t_forward,
            *torch.stack(Qzs).shape[1:],
            device=x.device,
            dtype=x.dtype
        )
        Qzs_sm[t_held_in - 1] = Qzs_filt[-1]

        # Start from the end and move backwards
        log_weights_backward = log_ws[-1]
        for t in range(t_held_in - 2, -1, -1):

            if marginal_smoothing:
                # Marginal smoothing, note this jas K^2 cost!

                prior_mean = self.rnn(
                    Qzs[t], noise_scale=0, u=vs[t - 1]
                )
                probs_ij = (
                    torch.distributions.Normal(
                        loc=prior_mean.unsqueeze(3), scale=eff_std_prior.unsqueeze(3)
                    )
                    .log_prob(Qzs_sm[t + 1])
                    .sum(axis=1)
                )
                log_denom = torch.logsumexp(
                    log_ws[t].unsqueeze(-1) * probs_ij[:, :], axis=1
                )

                log_weight = log_ws[t]
                for i in range(k):
                    log_nom_i = probs_ij[:, i]
                    reweight_i = torch.logsumexp(
                        log_weights_backward + log_nom_i - log_denom, axis=1
                    )
                    log_weight[:, i] += reweight_i
                indices = sample_indices_systematic(log_weight)
                Qzs_sm[t] = resample_Q(Qzs[t], indices)
            else:
                # Conditional smoothing
                prior_mean = self.rnn.transition(
                    Qzs[t].unsqueeze(-2), v=vs[t - 1].unsqueeze(-2)
                ).squeeze(-2)
                ll_pz = (
                    torch.distributions.Normal(loc=prior_mean, scale=eff_std_prior)
                    .log_prob(Qzs_sm[t + 1])
                    .sum(axis=1)
                )
                log_weights_reweighted = log_ws[t] + ll_pz

                # Resample based on the backward weights
                indices = sample_indices_systematic(log_weights_reweighted)
                Qzs_sm[t] = resample_Q(Qzs[t], indices)

        # Use forward samples for the last n_forward steps

        for t in range(t_held_in, t_held_in + t_forward):
            prior_mean = self.rnn.transition(Qz.unsqueeze(-2), v=v).squeeze(-2)
            if sim_v:
                v = self.rnn.transition.step_input(v, torch.zeros_like(v))
            vs.append(v.squeeze(2))

            Q_dist = torch.distributions.Normal(
                loc=prior_mean, scale=torch.sqrt(eff_var_prior)
            )
            Qz = Q_dist.rsample()
            Qzs_filt.append(Qz)
            Qzs_sm[t] = Qz

        Qzs_filt = torch.stack(Qzs_filt).permute(1, 2, 0, 3)
        Qzs_sm = Qzs_sm.permute(1, 2, 0, 3)
        vs = torch.stack(vs).permute(1, 2, 0, 3)
        Xs_filt = self.rnn.get_observation(Qzs_filt, noise_scale=0, v=vs)
        Xs_sm = self.rnn.get_observation(Qzs_sm, noise_scale=0, v=vs)

        return Qzs_filt, Qzs_sm, Xs_filt, Xs_sm


def resample_Q(Qz, indices):
    """
    Batch resample
    Args:
        Qz (torch.tensor; BS, dim_z, K): input data
        indices (torch.tensor; BS, K): indices to resample
    Returns:
        Qz_resampled (torch.tensor; BS, dim_z, K): resampled data
    """
    return torch.gather(Qz, 2, indices.unsqueeze(1).expand(Qz.shape))


def sample_indices_systematic(log_weight):
    """Sample ancestral index using systematic resampling.
    From: https://github.com/tuananhle7/aesmc

    Args:
        log_weight (torch.tensor; BS, K): log of unnormalized weights, tensor
    Returns:
        indices (torch.tensor; BS, K): sampled indices
    """

    if torch.sum(log_weight != log_weight).item() != 0:
        raise FloatingPointError("log_weight contains nan element(s)")

    batch_size, num_particles = log_weight.size()
    indices = torch.zeros(
        batch_size, num_particles, device=log_weight.device, dtype=torch.long
    )
    log_weight = log_weight.to(dtype=torch.double).detach()
    uniforms = torch.rand(
        size=[batch_size, 1], device=log_weight.device, dtype=log_weight.dtype
    )
    pos = (
        uniforms + torch.arange(0, num_particles, device=log_weight.device)
    ) / num_particles

    normalized_weights = torch.exp(
        log_weight - torch.logsumexp(log_weight, axis=1, keepdims=True)
    )

    cumulative_weights = torch.cumsum(normalized_weights, axis=1)
    # hack to prevent numerical issues
    max = torch.max(cumulative_weights, axis=1, keepdims=True).values
    cumulative_weights = cumulative_weights / max

    for batch in range(batch_size):
        indices[batch] = torch.bucketize(pos[batch], cumulative_weights[batch])

    return indices


def sample_indices_multinomial(log_w):
    """Sample ancestral index using multinomial resampling.
    Args:
        log_weight (torch.tensor; BS, K): log of unnormalized weights, tensor
    Returns:
        indices (torch.tensor; BS, K): sampled indices
    """
    k = log_w.shape[1]
    log_w_tilde = log_w - torch.logsumexp(log_w, dim=1, keepdim=True)
    w_tilde = log_w_tilde.exp().detach()  # +1e-5
    w_tilde = w_tilde / w_tilde.sum(1, keepdim=True)
    return torch.multinomial(w_tilde, k, replacement=True)  # m* numsamples


def norm_and_detach_weights(log_w):
    """
    Normalize and detach weights
    Args:
        log_weight (torch.tensor; BS, K): log of unnormalized weights, tensor
    Returns:
        reweight (torch.tensor; BS, K): normalized weights
    """
    log_weight = log_w.detach()
    log_weight = log_weight - torch.max(log_weight, -1, keepdim=True)[0]  #
    reweight = torch.exp(log_weight)
    reweight = reweight / torch.sum(reweight, -1, keepdim=True)
    return reweight
