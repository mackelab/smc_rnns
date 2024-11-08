import torch
import torch.nn as nn
import numpy as np
from initialize_parameterize import *
from torch.nn.utils.parametrizations import orthogonal


class LRRNN(nn.Module):
    """
    Low-rank RNN
    Code inspired by https://github.com/DurstewitzLab/dendPLRNN
    """

    def __init__(self, dim_x, dim_z, dim_u, dim_N, params):
        """
        Args:
            dim_x (int): dimensionality of the data
            dim_z (int): dimensionality of the latent space (rank)
            dim_u (int): dimensionality of the input
            dim_N (int): amount of neurons in the network
            params (dict): dictionary of parameters
        """

        super(LRRNN, self).__init__()
        self.d_x = dim_x
        self.d_z = dim_z
        self.d_u = dim_u
        self.d_N = dim_N

        self.params = params
        self.normal = torch.distributions.Normal(0, 1)

        # Initialise noise
        # ------

        # need to keep diag positive
        self.chol_cov_embed = lambda x: torch.tril(x, diagonal=-1) + torch.diag_embed(
            torch.exp(x[range(x.shape[0]), range(x.shape[0])] / 2)
        )
        self.full_cov_embed = lambda x: self.chol_cov_embed(x) @ (
            self.chol_cov_embed(x).T
        )

        # initialise the observation noise
        # only used with Gaussian observations
        if params["scalar_noise_x"] == "Cov":
            self.R_x = nn.Parameter(
                torch.eye(self.d_x) * np.log(params["init_noise_x"]) * 2,
                requires_grad=params["train_noise_x"],
            )
            self.std_embed_x = lambda x: torch.sqrt(
                torch.diagonal(self.full_cov_embed(x))
            )
        elif params["scalar_noise_x"]:
            self.R_x = nn.Parameter(
                torch.ones(1) * np.log(params["init_noise_x"]) * 2,
                requires_grad=params["train_noise_x"],
            )
            self.std_embed_x = lambda log_var: torch.exp(log_var / 2).expand(self.d_x)
            self.var_embed_x = lambda log_var: torch.exp(log_var).expand(self.d_x)
        else:
            self.R_x = nn.Parameter(
                torch.ones(self.d_x) * np.log(params["init_noise_x"]) * 2,
                requires_grad=params["train_noise_x"],
            )
            self.std_embed_x = lambda log_var: torch.exp(log_var / 2)
            self.var_embed_x = lambda log_var: torch.exp(log_var)

        # initialise the latent noise
        if params["scalar_noise_z"] == "Cov":
            self.R_z = nn.Parameter(
                torch.eye(self.d_z) * np.log(params["init_noise_z"]) * 2,
                requires_grad=params["train_noise_z"],
            )
            self.std_embed_z = lambda x: torch.sqrt(
                torch.diagonal(self.full_cov_embed(x))
            )
            self.var_embed_z = lambda x: (self.full_cov_embed(x))
        elif params["scalar_noise_z"]:
            self.R_z = nn.Parameter(
                torch.ones(1) * np.log(params["init_noise_z"]) * 2,
                requires_grad=params["train_noise_z"],
            )
            self.std_embed_z = lambda log_var: torch.exp(log_var / 2).expand(self.d_z)
            self.var_embed_z = lambda log_var: torch.exp(log_var).expand(self.d_z)
        else:
            self.R_z = nn.Parameter(
                torch.ones(self.d_z) * np.log(params["init_noise_z"]) * 2,
                requires_grad=params["train_noise_z"],
            )
            self.std_embed_z = lambda log_var: torch.exp(log_var / 2)
            self.var_embed_z = lambda log_var: torch.exp(log_var)

        #  initialise the latent noise for t = 0
        if params["scalar_noise_z_t0"] == "Cov":
            self.R_z_t0 = nn.Parameter(
                torch.eye(self.d_z) * np.log(params["init_noise_z"]) * 2,
                requires_grad=params["train_noise_z_t0"],
            )
            self.std_embed_z_t0 = lambda x: torch.sqrt(
                torch.diagonal(self.full_cov_embed(x))
            )
            self.var_embed_z_t0 = lambda x: (self.full_cov_embed(x))
        elif params["scalar_noise_z_t0"]:
            self.R_z_t0 = nn.Parameter(
                torch.ones(1) * np.log(params["init_noise_z_t0"]) * 2,
                requires_grad=params["train_noise_z_t0"],
            )
            self.std_embed_z_t0 = lambda log_var: torch.exp(log_var / 2).expand(
                self.d_z
            )
            self.var_embed_z_t0 = lambda log_var: torch.exp(log_var).expand(self.d_z)
        else:
            self.R_z_t0 = nn.Parameter(
                torch.ones(self.d_z) * np.log(params["init_noise_z_t0"]) * 2,
                requires_grad=params["train_noise_z_t0"],
            )
            self.std_embed_z_t0 = lambda log_var: torch.exp(log_var / 2)
            self.var_embed_z_t0 = lambda log_var: torch.exp(log_var)

        # initialise the transition step
        # ---------
        if "clipped" in params.keys():
            if params["clipped"] and params["activation"] == "relu":
                params["activation"] = "clipped_relu"

        if "full_rank" in params.keys() and params["full_rank"] == True:
            print("using full rank transitions")
            self.transition = Transition_FullRank(
                self.d_z,
                self.d_u,
                nonlinearity=params["activation"],
                exp_par=params["exp_par"],
                shared_tau=params["shared_tau"],
                g=params["g"],
                train_latent_bias=params["train_latent_bias"],
                train_neuron_bias=params["train_neuron_bias"],
            )
        else:
            self.transition = Transition(
                self.d_z,
                self.d_u,
                self.d_N,
                nonlinearity=params["activation"],
                exp_par=params["exp_par"],
                shared_tau=params["shared_tau"],
                weight_dist=params["weight_dist"],
                m_orth=params["orth"],
                weight_scaler=params["weight_scaler"],
                train_latent_bias=params["train_latent_bias"],
                train_neuron_bias=params["train_neuron_bias"],
            )

        # initialise the observation step
        # ---------

        self.readout_from = params["readout_from"]

        # initialise the observation step, either readout from the latent states (+ potentially input dynamics)
        # or from the neuron activity
        if self.readout_from == "rates" or self.readout_from == "currents":
            out_dim = self.d_N  # readout from N-dim neuron activity
        elif self.readout_from == "z_and_v":
            out_dim = self.d_z + self.d_u  # readout from latents and inputs
        else:
            out_dim = self.d_z  # readout from latents

        self.observation = Observation(
            out_dim,
            self.d_x,
            train_bias=params["train_obs_bias"],
            train_weights=params["train_obs_weights"],
            identity_readout=params["identity_readout"],
            out_nonlinearity=params["out_nonlinearity"],
        )

        # initialise the initial state
        # ---------
        if "full_rank" in params.keys() and params["full_rank"] == True:
            self.initial_state = nn.Parameter(torch.zeros(self.d_z), requires_grad=True)
            self.get_initial_state = lambda u: self.initial_state.unsqueeze(0)

        elif params["initial_state"] == "zero":
            self.initial_state = nn.Parameter(
                torch.zeros(self.d_z), requires_grad=False
            )
            self.get_initial_state = lambda u: self.initial_state.unsqueeze(
                0
            ) + orth_proj(
                self.transition.m_transform(self.transition.m),
                torch.einsum("Nu,Bu->BN", self.transition.Wu, u),
            )
        elif params["initial_state"] == "trainable":
            self.initial_state = nn.Parameter(torch.zeros(self.d_z), requires_grad=True)
            self.get_initial_state = lambda u: self.initial_state.unsqueeze(
                0
            ) + orth_proj(
                self.transition.m_transform(self.transition.m),
                torch.einsum("Nu,Bu->BN", self.transition.Wu, u),
            )

        elif params["initial_state"] == "bias":
            self.get_initial_state = lambda u: -self.transition.h.unsqueeze(
                0
            ) + orth_proj(
                self.transition.m_transform(self.transition.m),
                torch.einsum("Nu,Bu->BN", self.transition.Wu, u),
            )

    def forward(self, z, noise_scale=0, u=None, v=None, sim_v=False):
        """forward step of the RNN, predict z one step ahead
        Args:
            z (torch.tensor; n_trials x dim_z x time_steps x k): latent time series
            noise_scale (float): scale of the noise
            u (torch.tensor; n_trials x dim_u x time_steps x k): raw input
            v (torch.tensor; n_trials x dim_u x time_steps x k): input filtered by RNN dynamics
            sim_v (bool): whether to simulate input dynamics, set to true for time-varying inputs!

        Returns:
            z (torch.tensor; n_trials x dim_z x time_steps x k): latent time series
            v (torch.tensor; n_trials x dim_z x time_steps x k): input filtered by RNN dynamics
        """
        if u is not None and sim_v == False:
            v = u
        z = self.transition(z, v=v)
        if u is not None:
            v = self.transition.step_input(v, u)

        if noise_scale > 0:
            if self.params["scalar_noise_z"] == "Cov":
                cov_chol = self.chol_cov_embed(self.R_z)
                z += noise_scale * torch.einsum(
                    "xz, BzTK -> BxTK", cov_chol, self.normal.sample(z.shape)
                )
            else:
                z += (
                    noise_scale
                    * self.normal.sample(z.shape)
                    * self.std_embed_z(self.R_z).unsqueeze(0).unsqueeze(2).unsqueeze(3)
                )
        return z, v

    def get_latent_time_series(
        self, time_steps=1000, cut_off=0, noise_scale=1, z0=None, u=None, sim_v=False
    ):
        """
        Generate a latent time series of length time_steps
        Args:
            time_steps (int): length of the latent time series
            cut_off (int): cut off the first cut_off time steps
            noise_scale (float): scale of the noise
            z0 (torch.tensor; n_trials x dim_z x 1): initial latent state
            u (torch.tensor; n_trials x dim_u x time_steps): input
            sim_v (bool): whether to simulate input dynamics, set to true for time-varying inputs!
        Returns:
            Z (torch.tensor; n_trials x dim_z x time_steps x k): latent time series
        """
        with torch.no_grad():
            Z = []
            V = []
            if z0 is None:
                z = torch.randn(1, self.d_z, 1, 1, device=self.R_x.device)

            # set initial state
            else:
                if z0.shape[0] == 1 or len(z0.shape) == 1:  # only z dimension is given
                    z = z0.to(device=self.R_x.device).reshape(1, self.d_z, 1, 1)
                elif len(z0.shape) < 4:  # trial and z dimension is given
                    z = z0.to(device=self.R_x.device).reshape(
                        z0.shape[0], self.d_z, 1, 1
                    )
                else:
                    z = z0.to(device=self.R_x.device)

            # run model with input
            if u is not None:
                if len(u.shape) < 4:
                    u = u.unsqueeze(-1)  # add particle dim
                v = torch.zeros(u.shape[0], self.d_u, 1, 1, device=self.R_x.device)
                for t in range(time_steps + cut_off):

                    z, v = self.forward(
                        z,
                        noise_scale=noise_scale,
                        u=u[:, :, t].unsqueeze(2),
                        v=v,
                        sim_v=sim_v,
                    )
                    Z.append(z[:, :, 0])
                    V.append(v[:, :, 0])
                V = torch.stack(V)
                V = V[cut_off:]
                V = V.permute(1, 2, 0, 3)
            # run model without input
            else:
                print("no input")
                for t in range(time_steps + cut_off):
                    z, _ = self.forward(z, noise_scale=noise_scale)
                    Z.append(z[:, :, 0])

            # cut off the transients
            Z = torch.stack(Z)
            Z = Z[cut_off:]
            Z = Z.permute(1, 2, 0, 3)

        if sim_v:
            return Z, V
        return Z

    def get_rates(self, z, u=None):
        """transform the latent states to the neuron activity"""
        R = self.transition.get_rates(z, u=u)
        return R

    def get_observation(self, z, v=None, noise_scale=0):
        """
        Generate observations from the latent states
        Args:
            z (torch.tensor; n_trials x dim_z x time_steps x k): latent time series
            noise_scale (float): scale of the noise
            v (torch.tensor; n_trials x dim_u x time_steps x k): input filtered by RNN dynamics
        Returns:
            X (torch.tensor; n_trials x dim_x x time_steps x k): observations
        """
        if self.readout_from == "rates":
            R = self.get_rates(z, u=v)
        elif self.readout_from == "currents":
            m = self.transition.m_transform(self.transition.m)
            if v is not None:
                Wu = self.transition.Wu
                R = torch.einsum("Nz,BzTK->BNTK", m, z) + torch.einsum(
                    "Nz,BzTK->BNTK", Wu, v
                )
            else:
                R = torch.einsum("Nz,BzTK->BNTK", m, z)
        elif self.readout_from == "z_and_v":
            R = torch.concat((z, v.repeat(1, 1, 1, z.shape[-1])), dim=1)
        else:
            R = z
        X = self.observation(R)
        X += (
            noise_scale
            * self.normal.sample(X.shape)
            * self.std_embed_x(self.R_x).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        )
        return X

    def inv_observation(self, X, grad=True):
        """
        Args:
            X (torch.tensor; n_trials x dim_x x time_steps): observations
            grad (bool): whether to allow autodiff through the inversion
        Returns:
            z (torch.tensor; n_trials x dim_z x time_steps): latent time series
        """

        if self.readout_from == "currents":
            m = self.transition.m_transform(self.transition.m)
            Wu = self.transition.Wu
            mW = torch.cat((m, Wu), dim=1)
            B_inv = torch.linalg.pinv(
                (self.observation.cast_B(self.observation.B).T @ mW).T
            )[:, : self.d_z]
        else:
            B_inv = torch.linalg.pinv(self.observation.cast_B(self.observation.B))

        if grad:
            return torch.einsum(
                "xz,bxT->bzT", (B_inv, X - self.observation.Bias.squeeze(-1))
            )
        else:
            return torch.einsum(
                "xz,bxT->bzT",
                (B_inv.detach(), X - self.observation.Bias.squeeze(-1).detach()),
            )


class Observation(nn.Module):
    """
    Readout from the latent states or the neuron activity
    """

    def __init__(
        self,
        dy,
        dx,
        train_bias=True,
        train_weights=True,
        identity_readout=False,
        out_nonlinearity="identity",
    ):
        """
        Args:
            dy (int): dimensionality of the input to the readout (can be N, d_z, or d_z+d_u, depending on the readout_from)
            dx (int): dimensionality of the data
            train_bias (bool): whether to train the bias
            train_weights (bool): whether to train the weights
            identity_readout (bool): whether to use the identity matrix as the readout matrix, 
                                     - set to True if a one-to-one mapping between observed and
                                     RNN units is desired!
            out_nonlinearity (string): use e.g., 'softplus' to rectify rates for Poisson observations
        """
        super(Observation, self).__init__()
        self.dy = dy
        self.dx = dx

        if identity_readout:
            B = torch.zeros(self.dy, self.dx)
            B[range(self.dx), range(self.dx)] = 1
            self.B = nn.Parameter(B, requires_grad=train_weights)
            self.mask = B
            self.cast_B = lambda x: x * self.mask
        else:
            self.B = nn.Parameter(
                np.sqrt(2 / dy) * torch.randn(self.dy, self.dx),
                requires_grad=train_weights,
            )
            self.cast_B = lambda x: x
            self.mask = torch.ones(1)

        self.Bias = nn.Parameter(
            torch.zeros(1, self.dx, 1, 1), requires_grad=train_bias
        )

        # for Poisson we need to rectify outputs to be positive
        if out_nonlinearity == "exp":
            self.nonlinearity = torch.exp
        elif out_nonlinearity == "relu":
            self.nonlinearity = lambda x: torch.relu(x) + 1e-10
        elif out_nonlinearity == "softplus":
            self.nonlinearity = torch.nn.functional.softplus
        elif out_nonlinearity == "identity":
            self.nonlinearity = lambda x: x

    def forward(self, z):
        """
        Args:
            z (torch.tensor; n_trials x dim_z x time_steps x k): latent time series
        Returns:
            X (torch.tensor; n_trials x dim_x x time_steps x k): observations
        """
        return self.nonlinearity(
            torch.einsum("zx,bzTK->bxTK", (self.cast_B(self.B), z)) + self.Bias
        )


class Transition(nn.Module):
    """
    Latent dynamics of the prior, parameterised by a low-rank RNN
    """

    def __init__(
        self,
        dz,
        du,
        hidden_dim,
        nonlinearity,
        exp_par,
        shared_tau,
        weight_dist="uniform",
        m_orth=False,
        weight_scaler=1,
        train_latent_bias=True,
        train_neuron_bias=True,
    ):
        """
        Args:
            dz (int): dimensionality of the latent space
            hidden_dim (int): amount of neurons in the network
            nonlinearity (str): nonlinearity of the hidden layer
            exp_par (bool): whether to use the exponential parameterisation for time constants
            shared_tau (bool): whether to have one shared time constant across the latent dimensions
            weight_dist (str): weight distribution
            m_orth (bool): whether to orthogonalise the left singular vectors
            weight_scaler (float): scaling factor for the weights
            train_latent_bias (bool): whether to train the bias of the latents (z)
            train_neuron_bias (bool): whether to train the bias of the neurons (x)
        """
        super(Transition, self).__init__()
        self.dz = dz
        self.du = du

        # nonlinearity
        if nonlinearity == "relu":
            print("using ReLU activation")
            relu = torch.nn.ReLU()
            self.nonlinearity = lambda x, h: relu(x - h)
            self.dnonlinearity = relu_derivative
        elif nonlinearity == "clipped_relu":
            print("using clipped ReLU activation")
            relu = torch.nn.ReLU()
            self.nonlinearity = lambda x, h: relu(x + h) - relu(x)
            self.dnonlinearity = clipped_relu_derivative
        elif nonlinearity == "tanh":
            print("using tanh activation")
            self.nonlinearity = lambda x, h: torch.nn.Tanh(x - h)
            self.dnonlinearity = tanh_derivative
        elif nonlinearity == "identity":
            print("using identity activation")
            self.nonlinearity = lambda x, h: x - h
            self.dnonlinearity = lambda x: torch.ones_like(x)

        # time constants
        if shared_tau:
            if exp_par:
                self.AW = nn.Parameter(
                    torch.log(-torch.log(torch.ones(1, 1, 1, 1) * shared_tau))
                )
                self.cast_A = lambda x: torch.exp(-torch.exp(x))
            else:
                self.AW = nn.Parameter(torch.ones(1, 1, 1, 1) * shared_tau)
                self.cast_A = lambda x: x
        else:
            if exp_par:
                self.AW = init_AW_exp_par(self.dz)
                self.cast_A = exp_par_F
            else:
                self.AW = init_AW(self.dz)
                self.cast_A = lambda x: x.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        # bias of the neurons
        if nonlinearity == "clipped_relu":
            self.h = nn.Parameter(
                uniform_init1d(hidden_dim), requires_grad=train_neuron_bias
            )
        else:
            self.h = nn.Parameter(
                torch.zeros(hidden_dim), requires_grad=train_neuron_bias
            )

        # bias of the latents
        self.hz = nn.Parameter(torch.zeros(dz), requires_grad=train_latent_bias)

        # weights (left and right singular vectors)
        if not m_orth:
            if weight_dist == "uniform":
                self.n, self.m = initialize_Ws_uniform(dz, hidden_dim)
            elif weight_dist == "gauss":
                self.n, self.m = initialize_Ws_gauss(dz, hidden_dim)
            else:
                print("WARNING: weight distribution not implemented, using uniform")
                self.n, self.m = initialize_Ws_uniform(dz, hidden_dim)
            self.m_transform = lambda x: x

        else:
            print("orthogonalising m")
            # Orthonormal columns
            self.m = orthogonal(nn.Linear(dz, hidden_dim, bias=False))
            self.n, _ = initialize_Ws_uniform(dz, hidden_dim)
            self.m_transform = lambda x: x.weight
        self.scaling = weight_scaler
        print("weight scaler", self.scaling)

        # Input weights
        if self.du > 0:
            self.Wu = nn.Parameter(
                uniform_init2d(hidden_dim, self.du), requires_grad=True
            )
        else:
            self.Wu = torch.zeros(hidden_dim, 0)

    def forward(self, z, v=None):
        """
        Latent RNN (internal) dynamics, one step forward
        Args:
            z (torch.tensor; n_trials x dim_z x time_steps x k): latent time series
            v (torch.tensor; n_trials x dim_u x time_steps x k): filtered input 
        Returns:
            z (torch.tensor; n_trials x dim_z x time_steps x k): latent time series
        """
        A = self.cast_A(self.AW)
        R = self.get_rates(z, v=v)
        z = (
            A * z
            + torch.einsum("zN,BNTK->BzTK", self.n * self.scaling, R)
            + self.hz.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        )
        return z

    def step_input(self, v, u):
        """
        Latent RNN input dynamics, one step forward
        Args:
            v (torch.tensor; n_trials x dim_u x time_steps x k): input filtered by RNN dynamics
            u (torch.tensor; n_trials x dim_u x time_steps x k): raw input
        Returns:
            v (torch.tensor; n_trials x dim_u x time_steps x k): input filtered by RNN dynamics
        """
        A = self.cast_A(self.AW)
        v = A * v + (1 - A) * u
        return v

    def get_rates(self, z, v=None):
        """Transform latents to neuron activity
        Args:
            z (torch.tensor; n_trials x dim_z x time_steps x k): latent time series
            v (torch.tensor; n_trials x dim_u x time_steps x k): filtered input
        Returns:
            R (torch.tensor; n_trials x dim_N x time_steps x k): neuron activity"""
        m = self.m_transform(self.m)
        X = torch.einsum("Nz,BzTK->BNTK", m, z)
        if v is not None:
            X += torch.einsum("Nu,BuTK->BNTK", self.Wu, v)
        R = self.nonlinearity(X, self.h.unsqueeze(0).unsqueeze(2).unsqueeze(3))
        return R


class Transition_FullRank(nn.Module):
    """
    Alternative latent dynamics of the prior
    parameterised by a full-rank RNN
    """

    def __init__(
        self,
        dz,
        du,
        nonlinearity,
        exp_par,
        shared_tau,
        g=np.sqrt(2),
        train_latent_bias=True,
        train_neuron_bias=True,
    ):
        """
        Args:
            dz (int): dimensionality of the latent space
            hidden_dim (int): amount of neurons in the network
            nonlinearity (str): nonlinearity of the hidden layer
            exp_par (bool): whether to use the exponential parameterisation for time constants
            shared_tau (bool): whether to have one shared time constant across the latent dimensions
            weight_dist (str): weight distribution
            shared_tau (float): related to time constant tau as (1-dt/tau)
            g (float): scale/gain of the recurrent weights
            train_latent_bias (bool): whether to train the bias of the latents (z)
            train_neuron_bias (bool): whether to train the bias of the neurons (x)
        """
        super(Transition_FullRank, self).__init__()
        self.dz = dz
        self.du = du

        # nonlinearity
        if nonlinearity == "relu":
            print("using ReLU activation")
            relu = torch.nn.ReLU()
            self.nonlinearity = lambda x, h: relu(x - h)
            self.dnonlinearity = relu_derivative
        elif nonlinearity == "clipped_relu":
            print("using clipped ReLU activation")
            relu = torch.nn.ReLU()
            self.nonlinearity = lambda x, h: relu(x + h) - relu(x)
            self.dnonlinearity = clipped_relu_derivative
        elif nonlinearity == "tanh":
            print("using tanh activation")
            self.nonlinearity = lambda x, h: torch.nn.Tanh(x - h)
            self.dnonlinearity = tanh_derivative
        elif nonlinearity == "identity":
            print("using identity activation")
            self.nonlinearity = lambda x, h: x - h
            self.dnonlinearity = lambda x: torch.ones_like(x)

        # time constants
        if shared_tau:
            if exp_par:
                self.AW = nn.Parameter(
                    torch.log(-torch.log(torch.ones(1, 1, 1, 1) * shared_tau))
                )
                self.cast_A = lambda x: torch.exp(-torch.exp(x))
            else:
                self.AW = nn.Parameter(torch.ones(1, 1, 1, 1) * shared_tau)
                self.cast_A = lambda x: x
        else:
            if exp_par:
                self.AW = init_AW_exp_par(self.dz)
                self.cast_A = exp_par_F
            else:
                self.AW = init_AW(self.dz)
                self.cast_A = lambda x: x.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        # bias of the neurons
        if nonlinearity == "clipped_relu":
            self.h = nn.Parameter(uniform_init1d(dz), requires_grad=train_neuron_bias)
        else:
            self.h = nn.Parameter(torch.zeros(dz), requires_grad=train_neuron_bias)

        # bias of the latents
        self.hz = nn.Parameter(torch.zeros(dz), requires_grad=train_latent_bias)

        # weights (left and right singular vectors)
        self.W = nn.Parameter(torch.randn(dz, dz) * g / np.sqrt(dz), requires_grad=True)

        # Input weights
        if self.du > 0:
            self.Wu = nn.Parameter(uniform_init2d(dz, self.du), requires_grad=True)
        else:
            self.Wu = torch.zeros(dz, 0)

    def forward(self, z, v=None):
        """
        One step forward
        Args:
            z (torch.tensor; n_trials x dim_z x time_steps x k): latent time series
            u (torch.tensor; n_trials x dim_u x time_steps x k): input
        Returns:
            z (torch.tensor; n_trials x dim_z x time_steps x k): latent time series
        """
        A = self.cast_A(self.AW)
        z = (
            A * z
            + (1 - A)
            * (
                torch.einsum(
                    "zN,BNTK->BzTK",
                    self.W,
                    self.nonlinearity(z, self.h.unsqueeze(0).unsqueeze(2).unsqueeze(3)),
                )
            )
            + self.hz.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        )
        if v is not None:
            z += torch.einsum("Nu,BuTK->BNTK", self.Wu, v)
        return z

    def step_input(self, v, u):
        """
        no need to simulate input dynamics seperately for full rank RNNs, so just return v
        """
        return v
