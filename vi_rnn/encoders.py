import torch
import torch.nn as nn
import numpy as np
import sys, os

file_dir = str(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(file_dir + "/..")


class Inverse_Observation(nn.Module):
    """
    Invert the (linear) observation model to obtain e(z|x)
    """

    def __init__(self, dim_x, dim_z, params, inv_obs):
        """
        Args:
            dim_x (int): dimensionality of the data
            dim_z (int): dimensionality of the latent space
            params (dict): dictionary of parameters
            inv_obs (func): inverse observation model
        """

        super(Inverse_Observation, self).__init__()
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.params = params

        # initialise a distribution from which we can sample
        self.normal = torch.distributions.Normal(0, 1)
        self.logvar = nn.Parameter(
            2 * torch.log(torch.ones(self.dim_z) * params["init_scale"])
        )

        # how much of the input data (in time steps) gets cut off in the forward pass
        self.cut_len = 0
        self.mean = inv_obs

    def forward(self, x, k=1):
        """
        Forward pass of the MLP encoder
        Args:
            x (torch.tensor; batch_size x dim_x x dim_T): data
            k (int): number of particles to draw from the approximate posterior
        Returns:
            z (torch.tensor; batch_size x dim_z x dim_T x k): sampled latent variables
            mean (torch.tensor; batch_size x dim_z x dim_T x k): mean of the approximate posterior
            logvar (torch.tensor; batch_size x dim_z x dim_T x k): log variance of the approximate posterior
            eps_sample (torch.tensor; batch_size x dim_z x dim_T x k): sample from the standard normal distribution
        """

        mean = (
            self.mean(x, grad=self.params["obs_grad"]).unsqueeze(-1).repeat(1, 1, 1, k)
        )
        logvar = (
            self.logvar.unsqueeze(0)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .repeat(1, 1, mean.shape[2], k)
        )
        eps_sample = self.normal.sample(mean.shape)
        z = mean + torch.exp(logvar / 2) * eps_sample
        return z, mean, logvar, eps_sample


class CNN_encoder_causal(nn.Module):
    """
    This a CNN to parameterise e(z|x)
    """

    def __init__(self, dim_x, dim_z, params):
        """
        Args:
            dim_x (int): dimensionality of the data
            dim_z (int): dimensionality of the latent space
            params (dict): dictionary of parameters
        """
        super(CNN_encoder_causal, self).__init__()
        self.dim_x = dim_x
        self.dim_z = dim_z
        kernels = params["init_kernel_sizes"]
        n_channels = params["n_channels"]
        self.params = params

        # initialise the convolutions
        initial_convs = []
        initial_convs_std = []

        # make padding nn Module class for zero padding to the front of sequence
        class Pad(nn.Module):
            def __init__(self, padding, mode):
                super().__init__()
                self.pad = padding
                self.mode = mode

            def forward(self, x):
                x = torch.nn.functional.pad(x, self.pad, mode=self.mode)
                return x

        for i in range(len(kernels) - 1):

            # zero pad for causal convs
            pad = (kernels[i] - 1, 0, 0, 0)
            initial_convs.append(Pad(pad, mode=params["padding_mode"]))
            initial_convs.append(
                nn.Conv1d(
                    in_channels=n_channels[i - 1] if i > 0 else self.dim_x,
                    out_channels=n_channels[i],
                    kernel_size=kernels[i],
                    padding="valid",
                )
            )

            if params["nonlinearity"] == "leaky_relu":
                initial_convs.append(torch.nn.LeakyReLU(0.1))
                initial_convs_std.append(torch.nn.LeakyReLU(0.1))
            elif params["nonlinearity"] == "gelu":
                initial_convs.append(torch.nn.GELU())
                initial_convs_std.append(torch.nn.GELU())

        pad = (kernels[-1] - 1, 0, 0, 0)

        initial_convs.append(Pad(pad, mode=params["padding_mode"]))

        self.initial_stack = nn.Sequential(*initial_convs)

        # and finally a seprate convolution to get the mean and covariance
        self.mean_conv = nn.Conv1d(
            in_channels=n_channels[-1],
            out_channels=self.dim_z,
            kernel_size=kernels[-1],
            padding="valid",
        )
        if params["constant_var"]:
            self.logvar = nn.Parameter(
                2 * torch.log(torch.ones(self.dim_z) * params["init_scale"])
            )
        else:
            self.logvar_conv = nn.Conv1d(
                in_channels=n_channels[-1],
                out_channels=self.dim_z,
                kernel_size=kernels[-1],
                padding="valid",
            )

            with torch.no_grad():
                self.logvar_conv.bias.copy_(
                    self.logvar_conv.bias + (np.log(params["init_scale"]) * 2)
                )  # = torch.zeros_like(self.logstd.bias)
            self.logvar = self.logvar_conv.bias

        # initialise a distribution from which we can sample
        self.normal = torch.distributions.Normal(0, 1)

        # how much of the input data (in time steps) gets cut off in the forward pass
        self.cut_len = 0
        print("cut_len: " + str(self.cut_len))

    def forward(self, x, k=1):
        """
        Forward pass of the CNN encoder
        Args:
            x (torch.tensor; batch_size x dim_x x dim_T): data
            k (int): number of particles to draw from the approximate posterior
        Returns:
            z (torch.tensor; batch_size x dim_z x dim_T x k): sampled latent variables
            mean (torch.tensor; batch_size x dim_z x dim_T x k): mean of the approximate posterior
            logvar (torch.tensor; batch_size x dim_z x dim_T x k): log variance of the approximate posterior
            eps_sample (torch.tensor; batch_size x dim_z x dim_T x k): sample from the standard normal distribution
        """
        init = self.initial_stack(x)
        mean = self.mean_conv(init)
        mean = mean.unsqueeze(-1).repeat(1, 1, 1, k)
        if self.params["constant_var"]:
            logvar = (
                self.logvar.unsqueeze(0)
                .unsqueeze(-1)
                .unsqueeze(-1)
                .repeat(1, 1, mean.shape[2], k)
            )
        else:
            logvar = self.logvar_conv(init).unsqueeze(-1).repeat(1, 1, 1, k)
        eps_sample = self.normal.sample(mean.shape)
        z = mean + torch.exp(logvar / 2) * eps_sample
        return z, mean, logvar, eps_sample