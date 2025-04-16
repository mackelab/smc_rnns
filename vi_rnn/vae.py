import os
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
import torch.nn as nn
from encoders import CNN_encoder
from vi_rnn.rnn import RNN


class VAE(nn.Module):
    """
    VAE with low-rank RNN / dynamical systems prior
    """

    def __init__(self, vae_params):
        """initialize"""

        super(VAE, self).__init__()

        # data in dimensionality
        self.dim_x = vae_params["dim_x"]

        # data out dimensionality (could be different e.g., when co-smoothing)
        if "dim_x_hat" in vae_params:
            self.dim_x_hat = vae_params["dim_x_hat"]
        else:
            self.dim_x_hat = vae_params["dim_x"]

        # input dimensionality
        if "dim_u" in vae_params:
            self.dim_u = vae_params["dim_u"]
        else:
            self.dim_u = 0

        # latent dimensionality
        self.dim_z = vae_params["dim_z"]

        # number of units in RNN
        self.dim_N = vae_params["dim_N"]

        self.vae_params = vae_params

        # Initialise RNN
        self.rnn = RNN(
            self.dim_x_hat,
            self.dim_z,
            self.dim_u,
            self.dim_N,
            vae_params["rnn_params"],
        )

        # Initialise encoder
        self.has_encoder = True
        if "enc_architecture" in vae_params:
            if vae_params["enc_architecture"] == "CNN":
                self.encoder = CNN_encoder(
                    self.dim_x, self.dim_z, vae_params["enc_params"]
                )
            else:
                print("Warning: encoder not recognised, continuing without encoder")
                self.has_encoder = False
        else:
            print("Initialising VAE without encoder")
            self.has_encoder = False

        # clamp variances
        self.min_var = 1e-8
        self.max_var = 100

    def to_device(self, device):
        """Move network between cpu / gpu (cuda)"""
        self.rnn.to(device=device)
        self.rnn.normal.loc = self.rnn.normal.loc.to(device=device)
        self.rnn.normal.scale = self.rnn.normal.scale.to(device=device)
        self.rnn.transition.Wu = self.rnn.transition.Wu.to(device=device)
        if self.has_encoder:
            self.encoder.to(device=device)
