import torch
import torch.nn as nn
import numpy as np
from initialize_parameterize import *
from torch.nn.utils.parametrizations import orthogonal


class RNN(nn.Module):
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

        super(RNN, self).__init__()
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u
        self.dim_N = dim_N

        self.params = params
        self.normal = torch.distributions.Normal(0, 1)

        # Initialise noise
        # ------

        # Gaussian observations
        if params['obs_likelihood'] == "Gauss":
            self.observation_distribution=lambda x, noise_scale=1:torch.distributions.Normal(loc=x, scale=self.std_embed_x(self.R_x).view(1,self.dim_x,*([1]*len(x.shape[2:])))*noise_scale)
        if "noise_x" in params.keys():
            self.R_x, self.std_embed_x, self.var_embed_x = init_noise(
                params["noise_x"], self.dim_x, params["init_noise_x"], params["train_noise_x"]
            )
      
        # Poisson observations
        elif params['obs_likelihood'] == "Poisson":
            self.observation_distribution=lambda x, noise_scale=None:torch.distributions.Poisson(x)
        else:
            raise ValueError("observation_likelihood not recognised, use Gauss or Poisson")       
        
        # sampling and likelihood functions
        self.get_observation_log_likelihood = lambda x_hat, x,noise_scale=1: self.observation_distribution(x,noise_scale=noise_scale).log_prob(x_hat).sum(axis=1)
        self.get_observation_sample = lambda x,noise_scale=1: self.observation_distribution(x,noise_scale).sample()
        self.obs_likelihood=params['obs_likelihood']
       
    
        # Latent states transition noise
        self.R_z, self.std_embed_z, self.var_embed_z = init_noise(
            params["noise_z"], self.dim_z, params["init_noise_z"], params["train_noise_z"]
        )

        # Initial latent state noise
        self.R_z_t0, self.std_embed_z_t0, self.var_embed_z_t0 = init_noise(
            params["noise_z_t0"],
            self.dim_z,
            params["init_noise_z_t0"],
            params["train_noise_z_t0"],
        )

        # initialise the transition step
        # ---------
        if params["transition"]=="low_rank":
            self.transition = Transition_LowRank(
                self.dim_z,
                self.dim_u,
                self.dim_N,
                nonlinearity=params["activation"],
                decay=params["decay"],
                weight_dist=params["weight_dist"],
                train_neuron_bias=params["train_neuron_bias"],
            )

        elif params["transition"]=="full_rank":
            self.transition = Transition_FullRank(
                self.dim_z,
                self.dim_u,
                nonlinearity=params["activation"],
                decay=params["decay"],
                g=params["g"],
                train_neuron_bias=params["train_neuron_bias"],
            )
        else:
            raise ValueError("transition not recognised, use low_rank or full_rank")
        
        # initialise the observation step
        # ---------
        self.readout_from = params["readout_from"]

        if params["observation"]=="one_to_one":
            if self.readout_from  == "rates":
                z_to_x_func=self.transition.get_rates
            elif self.readout_from  =="currents":
                z_to_x_func=self.transition.get_currents       
            else:
                raise ValueError("readout_from not recognised, use rates, currents (for a one_to_one obervation model)")
            self.observation = One_to_One_observation(
                    dim_x=self.dim_x,
                    z_to_x_func =z_to_x_func,
                    train_bias=params["train_obs_bias"],
                    train_weights=params["train_obs_weights"],
                    obs_nonlinearity=params["obs_nonlinearity"]
                )
        elif params["observation"]=="affine":
            if self.readout_from == "z_andim_v":
                dim_v=self.dim_u
            elif self.readout_from == "z":
                dim_v=0
            else:
                raise ValueError(
                    "readout_from not recognised, use z_andim_v, or z (for an affine observation model)"
                )
            self.observation = Affine_observation(
                    dim_x=self.dim_x,
                    dim_z = self.dim_z,
                    dim_v = dim_v,
                    train_bias=params["train_obs_bias"],
                    train_weights=params["train_obs_weights"],
                    obs_nonlinearity=params["obs_nonlinearity"]
                )
            
        self.simulate_input=params["simulate_input"]


      

        # initialise the initial state
        # ---------

        if params["transition"]=="full_rank":
            self.initial_state = nn.Parameter(torch.zeros(self.dim_z), requires_grad=True)
            self.get_initial_state = lambda u: self.initial_state.unsqueeze(0)

        elif params["initial_state"] == "zero":
            self.initial_state = nn.Parameter(
                torch.zeros(self.dim_z), requires_grad=False
            )
            self.get_initial_state = lambda u: self.initial_state.unsqueeze(
                0
            ) + orth_proj(
                self.transition.m,
                torch.einsum("Nu,Bu->BN", self.transition.Wu, u),
            )

        elif params["initial_state"] == "trainable":
            self.initial_state = nn.Parameter(torch.zeros(self.dim_z), requires_grad=True)
            self.get_initial_state = lambda u: self.initial_state.unsqueeze(
                0
            ) + orth_proj(
                self.transition.m,
                torch.einsum("Nu,Bu->BN", self.transition.Wu, u),
            )

        elif params["initial_state"] == "bias":
            self.get_initial_state = lambda u: -self.transition.h.unsqueeze(
                0
            ) + orth_proj(
                self.transition.m,
                torch.einsum("Nu,Bu->BN", self.transition.Wu, u),
            )



    def get_latent_time_series(
        self, time_steps=1000, cut_off=0, noise_scale=1, z0=None, u=None, k=1
    ):
        """
        Generate a latent time series of length time_steps
        Args:
            time_steps (int): length of the latent time series
            cut_off (int): cut off the first cut_off time steps
            noise_scale (float): optional scale of the standard deviation
            z0 (torch.tensor; n_trials x dim_z x 1): initial latent state
            u (torch.tensor; n_trials x dim_u x time_steps): input
            k (int): number of particles
        Returns:
            Z (torch.tensor; n_trials x dim_z x time_steps x k): latent time series
            V (torch.tensor; n_trials x dim_u x time_steps x k): processed input
        """
        with torch.no_grad():
            Z = []
            V = []
            
            if len(u.shape) < 4:
                u = u.unsqueeze(-1)  # add particle dim
            if u is None:
                u = torch.zeros(z0.shape[0],self.du,time_steps, 1)
            
            # get initial input
            if self.simulate_input:
                v = torch.zeros(u.shape[0], self.dim_u, 1, device=self.R_z.device)
            else:
                v = u[:, :, 0]

            # get initial state and add particle dim
            if z0 is None:
                z = (
                        self.rnn.get_initial_state(v[:, :, 0]).unsqueeze(2)
                        .expand(1, self.dim_z, k)
                    )
            else:
                if z0.shape[0] == 1 or len(z0.shape) == 1:  # only z dimension is given
                    z = z0.to(device=self.R_z.device).reshape(1, self.dim_z, 1).expand(1,self.dim_z,k)
                elif len(z0.shape) < 3:  # trial and z dimension is given
                    z = z0.to(device=self.R_z.device).reshape(
                        z0.shape[0], self.dim_z, 1
                    ).expand(z0.shape[0], self.dim_z,k)
                else:
                    z = z0.to(device=self.R_z.device).expand(z0.shape[0], self.dim_z,k)

            Z.append(z)
            V.append(v)

            for t in range(1,time_steps + cut_off):
                
                _, z = self.get_latent(z,v, noise_scale=noise_scale)

                # process input
                if self.simulate_input:
                    v = self.transition.step_input(v, u[:, :, t - 1])
                else:
                    v = u[:, :, t]

                Z.append(z)
                V.append(v)
            
            V = torch.stack(V)
            V = V[cut_off:]
            V = V.permute(1, 2, 0, 3)
            Z = torch.stack(Z)
            Z = Z[cut_off:]
            Z = Z.permute(1, 2, 0, 3)

        return Z, V

    def get_latent_sample(self, z, noise_scale=0):
        """sample latent given mean at current timestep
        Args:
            z (torch.tensor; n_trials x dim_z x time_steps x k): mean at time t
            noise_scale (float): optional scale of the standard deviation
        Returns:
            z_sample (torch.tensor; n_trials x dim_z x time_steps x k): sample at time t
        """
    
        if self.params["noise_z"] == "full":
            cov_chol = chol_cov_embed(self.R_z)
            z_sample = z + noise_scale * torch.einsum(
                "xz, Bz... -> Bx...", cov_chol, self.normal.sample(z.shape)
            )
        else:
            z_sample = z+ (
                noise_scale
                * self.normal.sample(z.shape)
                * self.std_embed_z(self.R_z).view(1,-1,1)
            )
        return z_sample

    def get_latent(self, z, v, noise_scale=0):
        """sample and mean given z at previous timestep
        Args:
            z (torch.tensor; n_trials x dim_z x time_steps x k): z at time t-1
            v (torch.tensor; n_trials x dim_u x time_steps x k): input
            noise_scale (float): optional scale of the standard deviation

        Returns:
            z_mean (torch.tensor; n_trials x dim_z x time_steps x k): mean at time t
            z_sample (torch.tensor; n_trials x dim_z x time_steps x k): sample at time t

        """
        z_mean = self.transition(z,v=v)
        z_sample = self.get_latent_sample(z_mean,noise_scale=noise_scale)
        return z_mean, z_sample
    
    def get_observation(self, z, v=None, noise_scale=1):
        """
        Generate observations from the latent states
        Args:
            z (torch.tensor; n_trials x dim_z x time_steps x k): latent time series
            v (torch.tensor; n_trials x dim_u x time_steps x k): input filtered by RNN dynamics
            noise_scale (float): optional scale of the standard deviation

        Returns:
            X_mean (torch.tensor; n_trials x dim_x x time_steps x k): observations mean
            X_sample (torch.tensor; n_trials x dim_x x time_steps x k): observations sample

        """
        X_mean = self.observation(z,v)
        X_sample = self.get_observation_sample(X_mean,noise_scale=noise_scale)
        return X_mean,X_sample

class One_to_One_observation(nn.Module):
    """
    Readout from the the activity of neurons in the network
    """
    def __init__(
        self,
        dim_x,
        z_to_x_func,
        train_bias=True,
        train_weights=True,
        obs_nonlinearity="identity",
    ):
        """
        Args:
            dim_x (int): dimensionality of the data
            z_to_x_func: maps latents to RNN unit space
            train_bias (bool): whether to train the bias
            train_weights (bool): whether to train the weights
            obs_nonlinearity (string): use e.g., 'softplus' to rectify rates for Poisson observations
        """
        super(One_to_One_observation, self).__init__()
        self.dim_x =dim_x
        self.z_to_x_func = z_to_x_func 
        self.B = nn.Parameter(
            torch.ones(self.dim_x),
            requires_grad=train_weights,
        )

        self.Bias = nn.Parameter(
            torch.zeros(self.dim_x), requires_grad=train_bias
        )

        # for Poisson we need to rectify outputs to be positive
        if obs_nonlinearity == "exp":
            exp = torch.exp
            self.nonlinearity = lambda x: exp(x) + 1e-10
        elif obs_nonlinearity == "relu":
            self.nonlinearity = lambda x: torch.relu(x) + 1e-10
        elif obs_nonlinearity == "softplus":
            sp = torch.nn.functional.softplus
            self.nonlinearity = lambda x: sp(x) + 1e-6
        elif obs_nonlinearity == "identity":
            self.nonlinearity = lambda x: x
        else:
            raise ValueError(
                "obs_nonlinearity not recognised, use exp, relu, softplus, or identity"
            )    

    def forward(self, z, v):
        """
        Args:
            z (torch.tensor; n_trials x dim_z x k): latent time series
        Returns:
            X (torch.tensor; n_trials x dim_x x k): observations
        """
        x = self.z_to_x_func(z,v)
        bias = self.Bias.view(1,-1,*([1]*len(z.shape[2:])))
        B = self.B.view(1,-1,*([1]*len(z.shape[2:])))
        return self.nonlinearity(B*x+bias)
    
class Affine_observation(nn.Module):
    """
    Readout from the latent states
    """
    def __init__(
        self,
        dim_x,
        dim_z,
        dim_v=0,
        train_bias=True,
        train_weights=True,
        obs_nonlinearity = "identity"
    ):
        """
        Args:
            dim_x (int): dimensionality of the data
            dim_z (int): dimensionality of the latents
            dim_v (int): dimensionality of the input
            train_bias (bool): whether to train the bias
            train_weights (bool): whether to train the weights
            obs_nonlinearity (string): use e.g., 'softplus' to rectify rates for Poisson observations
        """
        super(Affine_observation, self).__init__()
        self.dim_x = dim_x
        self.dim_z= dim_z
        self.dim_v = dim_v

        self.B = nn.Parameter(
            np.sqrt(2 / (self.dim_z+self.dim_v)) * torch.randn(self.dim_z+self.dim_v, self.dim_x),
            requires_grad=train_weights,
        )

        self.Bias = nn.Parameter(
            torch.zeros(self.dim_x), requires_grad=train_bias
        )

        # for Poisson we need to rectify outputs to be positive
        if obs_nonlinearity == "exp":
            exp = torch.exp
            self.nonlinearity = lambda x: exp(x) + 1e-10
        elif obs_nonlinearity == "relu":
            self.nonlinearity = lambda x: torch.relu(x) + 1e-10
        elif obs_nonlinearity == "softplus":
            sp = torch.nn.functional.softplus
            self.nonlinearity = lambda x: sp(x) + 1e-6
        elif obs_nonlinearity == "identity":
            self.nonlinearity = lambda x: x
        else:
            raise ValueError(
                "obs_nonlinearity not recognised, use exp, relu, softplus, or identity"
            )
        
        # readout from z_andim_v
        if self.dim_v>0:
            self.cat_zv =  lambda z, v:  torch.concat([(v.repeat(*([1]*len(v.shape[:-1])), z.shape[-1])),z],dim=1)
        # or just z
        else:
            self.cat_zv = lambda z, v : z
        
    def forward(self, z, v):
        """
        Args:
            z (torch.tensor; n_trials x dim_z x time_steps x k): latent time series
        Returns:
            X (torch.tensor; n_trials x dim_x x time_steps x k): observations
        """
        zv = self.cat_zv(z,v)
        bias = self.Bias.view(1,-1,*([1]*len(z.shape[2:])))
        return self.nonlinearity(
            torch.einsum("zx,bz...->bx...", (self.B, zv))+bias
        )


class Transition_LowRank(nn.Module):
    """
    Latent dynamics of the prior, parameterised by a low-rank RNN
    """

    def __init__(
        self,
        dz,
        du,
        hidden_dim,
        nonlinearity,
        decay,
        weight_dist="uniform",
        train_neuron_bias=True,
    ):
        """
        Args:
            dz (int): dimensionality of the latent space
            du (int): dimensionality of the inputs
            hidden_dim (int): amount of neurons in the low-rank RNN
            nonlinearity (str): nonlinearity of the hidden layer
            decay (float): decay constant
            weight_dist (str): weight distribution
            train_neuron_bias (bool): whether to train the bias of the neurons (x)
        """
        super(Transition_LowRank, self).__init__()
        self.dz = dz
        self.du = du

        # nonlinearity
        if nonlinearity == "relu":
            relu = torch.nn.ReLU()
            self.nonlinearity = lambda x, h: relu(x - h)
            self.dnonlinearity = relu_derivative
        elif nonlinearity == "clipped_relu":
            relu = torch.nn.ReLU()
            self.nonlinearity = lambda x, h: relu(x + h) - relu(x)
            self.dnonlinearity = clipped_relu_derivative
        elif nonlinearity == "tanh":
            self.nonlinearity = lambda x, h: torch.nn.Tanh(x - h)
            self.dnonlinearity = tanh_derivative
        elif nonlinearity == "identity":
            self.nonlinearity = lambda x, h: x - h
            self.dnonlinearity = lambda x: torch.ones_like(x)
        else:
            raise ValueError(
                "nonlinearity not recognised, use relu, clipped_relu, tanh, or identity"
            )   
        # time constants
        self.decay_param = nn.Parameter(torch.log(-torch.log(torch.ones(1) * decay)))

        # bias of the neurons
        if nonlinearity == "clipped_relu":
            self.h = nn.Parameter(
                uniform_init1d(hidden_dim), requires_grad=train_neuron_bias
            )
        else:
            self.h = nn.Parameter(
                torch.zeros(hidden_dim), requires_grad=train_neuron_bias
            )

        # weights (left and right singular vectors)
        if weight_dist == "uniform":
            self.n, self.m = initialize_Ws_uniform(dz, hidden_dim)
        elif weight_dist == "gauss":
            self.n, self.m = initialize_Ws_gauss(dz, hidden_dim)
        else:
            print("WARNING: weight distribution not implemented, using uniform")
            self.n, self.m = initialize_Ws_uniform(dz, hidden_dim)

        # Input weights
        if self.du > 0:
            self.Wu = nn.Parameter(
                uniform_init2d(hidden_dim, self.du), requires_grad=True
            )
        else:
            self.Wu = torch.zeros(hidden_dim, 0)
            
    @property
    def decay(self):
        return torch.exp(-torch.exp(self.decay_param)).view(1,1,1)
    
    def forward(self, z, v=0):
        """
        Latent RNN (internal) dynamics, one step forward
        Args:
            z (torch.tensor; n_trials x dim_z x time_steps x k): latent time series
            v (torch.tensor; n_trials x dim_u x time_steps x k): filtered input
        Returns:
            z (torch.tensor; n_trials x dim_z x time_steps x k): latent time series
        """
        R = self.get_rates(z, v=v)
        z = self.decay * z + torch.einsum("zN,BN...->Bz...", self.n, R)
        return z

    def step_input(self, v, u):
        """
        Latent RNN input dynamics, one step forward
        Args:
            v (torch.tensor; n_trials x dim_u x k): input filtered by RNN dynamics
            u (torch.tensor; n_trials x dim_u x k): raw input
        Returns:
            v (torch.tensor; n_trials x dim_u x k): input filtered by RNN dynamics
        """
        v = self.decay * v + (1 - self.decay) * u
        return v

    def get_rates(self, z, v=0):
        """Transform latents to neuron activity, after nonlinearity
        Args:
            z (torch.tensor; n_trials x dim_z x k): latent time series
            v (torch.tensor; n_trials x dim_u x k): filtered input
        Returns:
            R (torch.tensor; n_trials x dim_N x k): neuron activity after nonlinearity"""
        X = self.get_currents(z,v)
        R = self.nonlinearity(X, self.h.view(1,-1,1))
        return R
    
    def get_currents(self, z, v=0):
        """Transform latents to neuron activity, before nonlinearity
        Args:
            z (torch.tensor; n_trials x dim_z x time_steps x k): latent time series
            v (torch.tensor; n_trials x dim_u x time_steps x k): filtered input
        Returns:
            X (torch.tensor; n_trials x dim_N x time_steps x k): neuron activity before nonlinearity"""
        X = torch.einsum("Nz,Bz...->BN...", self.m, z) + torch.einsum(
            "Nu,Bu...->BN...", self.Wu, v
        )
        return X


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
        decay,
        g=np.sqrt(2),
        train_neuron_bias=True,
    ):
        """
        Args:
            dz (int): mount of neurons in the full rank RNN
            du (int): dimensionality of the inputs
            nonlinearity (str): nonlinearity of the hidden layer
            decay (float): related to time constant tau as (1-dt/tau)
            g (float): scale/gain of the recurrent weights
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
        else:
            raise ValueError(
                "nonlinearity not recognised, use relu, clipped_relu, tanh, or identity"
            )   
        # time constants
        self.decay_param = nn.Parameter(torch.log(-torch.log(torch.ones(1) * decay)))

        # bias of the neurons
        if nonlinearity == "clipped_relu":
            self.h = nn.Parameter(uniform_init1d(dz), requires_grad=train_neuron_bias)
        else:
            self.h = nn.Parameter(torch.zeros(dz), requires_grad=train_neuron_bias)

        # weights (left and right singular vectors)
        self.W = nn.Parameter((1 - decay) * torch.randn(dz, dz) * g / np.sqrt(dz), requires_grad=True)

        # Input weights
        if self.du > 0:
            self.Wu = nn.Parameter(uniform_init2d(dz, self.du), requires_grad=True)
        else:
            self.Wu = torch.zeros(dz, 0)
    
    @property
    def decay(self):
        return torch.exp(-torch.exp(self.decay_param)).view(1,1,1)


    def forward(self, z, v=0):
        """
        One step forward
        Args:
            z (torch.tensor; n_trials x dim_z x time_steps x k): latent time series
            u (torch.tensor; n_trials x dim_u x time_steps x k): input
        Returns:
            z (torch.tensor; n_trials x dim_z x time_steps x k): latent time series
        """
        z = self.decay * z +  (
            torch.einsum(
                "zN,BN...->Bz...",
                self.W,
                self.nonlinearity(z, self.h.view(1,-1,1)),
            )
        )

        z += torch.einsum("Nu,Bu...->BN...", self.Wu, v)
        return z

    def step_input(self, v, u):
        """
        no need to simulate input dynamics seperately for full rank RNNs, so just return v
        """
        return v
