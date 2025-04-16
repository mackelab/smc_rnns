import numpy as np
import sys
import os

vi_rnn_dir = os.path.dirname(os.path.abspath(__file__)) + "/../.."
sys.path.append(vi_rnn_dir)
from vi_rnn.vae import VAE
from vi_rnn.train import train_VAE
from vi_rnn.datasets import Basic_dataset
from pathlib import Path
import wandb
import torch

sync_wandb = True  # whether to sync with wandb
data_path = vi_rnn_dir + "/data_untracked/train_hpc2.npy"
out_dir = vi_rnn_dir + "/models/hpc2/"
if sync_wandb:
    wandb.init()

"""
Runs a sweep over models as initialsed by WandB
"""
dim_N = 512  # number of RNN units
cuda = True  # whether to use cuda

# initialise encoder
enc_params = {
    "init_kernel_sizes": [24, 11, 1],
    "padding_mode": "constant",
    "nonlinearity": "gelu",
    "n_channels": [128, 64],
    "init_scale": 0.1,
    "constant_var": False,
    "padding_location": "causal",
}

enc_params["first_layer"] = wandb.config.first_layer
enc_params["init_kernel_sizes"] = [enc_params["first_layer"], 11, 1]

# initialise prior
rnn_params = {
    "transition": "low_rank",
    "observation": "affine",
    "train_noise_z": True,
    "train_noise_z_t0": True,
    "init_noise_z": 0.1,
    "init_noise_z_t0": 0.1,
    "noise_z": "diag",
    "noise_z_t0": "diag",
    "identity_readout": False,
    "activation": "clipped_relu",
    "decay": 0.9,
    "readout_from": "z",
    "train_obs_bias": True,
    "train_obs_weights": True,
    "train_neuron_bias": True,
    "weight_dist": "uniform",
    "initial_state": "trainable",
    "obs_nonlinearity": "softplus",
    "obs_likelihood": "Poisson",
    "simulate_input": False,
}

# initialise training parameters
training_params = {
    "smooth_at_eval": False,
    "init_state_eval": "posterior_sample",
    "run_eval": True,
    "t_forward": 0,
    "lr": 1e-3,
    "lr_end": 1e-6,
    "n_epochs": None,
    "grad_norm": 0,
    "eval_epochs": 100,
    "batch_size": 64,
    "cuda": cuda,
    "smoothing": 20,
    "freq_cut_off": 10000,
    "k": 64,
    "loss_f": "smc",
    "resample": "systematic",  # , multinomial or none"
}

training_params["n_epochs"] = wandb.config.n_epochs
training_params["batch_size"] = wandb.config.batch_size

task_params = {
    "dur": 100,
    "n_trials": 3000,
    "name": "",
    "dataset_name": data_path,
}

loaded_binary_matrix = np.load(task_params["dataset_name"])

truncated_spk = loaded_binary_matrix[:, :]
data_eval = np.load(task_params["dataset_name"])
task = Basic_dataset(
    task_params,
    data=truncated_spk.astype(np.float32),
    data_eval=data_eval.astype(np.float32),
)
dim_x = task.data.shape[0]
dim_z = wandb.config.rank

# initialise VAE

VAE_params = {
    "dim_x": dim_x,
    "dim_z": dim_z,
    "dim_N": dim_N,
    "enc_architecture": "CNN",
    "enc_params": enc_params,
    "rnn_params": rnn_params,
}

vae = VAE(VAE_params)
train_VAE(vae, training_params, task, sync_wandb=wandb, out_dir=out_dir, fname=None)
