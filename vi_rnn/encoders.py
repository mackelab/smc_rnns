import torch
import torch.nn as nn
import numpy as np
import sys, os

file_dir = str(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(file_dir + "/..")


class CNN_encoder(nn.Module):
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
        super(CNN_encoder, self).__init__()
        self.dim_x = dim_x
        self.dim_z = dim_z
        kernels = (
            params["init_kernel_sizes"]
            if "init_kernel_sizes" in params
            else params["kernel_sizes"]
        )
        n_channels = params["n_channels"]
        self.params = params
        print(
            "using "
            + params["padding_location"]
            + " "
            + params["padding_mode"]
            + " padding"
        )

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

            if params["padding_location"] == "causal":
                # zero pad for causal convs
                pad = (kernels[i] - 1, 0, 0, 0)
            elif params["padding_location"] == "acausal":
                # zero pad for acausal convs
                pad = (0, kernels[i] - 1, 0, 0)
            elif params["padding_location"] == "windowed":
                # pad for windowed convs
                pad = (kernels[i] // 2, (kernels[i] // 2) - 1, 0, 0)
            else:
                raise ValueError(
                    "padding_location not recognised, use 'causal', 'acausal' or 'windowed'"
                )

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
            else:
                raise ValueError(
                    "nonlinearity not recognised, use 'leaky_relu' or 'gelu'"
                )

            if params["padding_location"] == "causal":
                # zero pad for causal convs
                pad = (kernels[-1] - 1, 0, 0, 0)
            elif params["padding_location"] == "acausal":
                # zero pad for acausal convs
                pad = (0, kernels[-1] - 1, 0, 0)
            elif params["padding_location"] == "windowed":
                # pad for windowed convs
                pad = (kernels[-1] // 2, (kernels[-1] // 2) - 1, 0, 0)

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
        return mean, logvar
