import numpy as np
import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(1, str(Path(__file__).absolute().parent.parent.parent))
from vi_rnn.vae import VAE
from vi_rnn.train import train_VAE
from vi_rnn.datasets import Oscillations_Cont
from vi_rnn.utils import *
from py_rnn.train import load_rnn

import datetime
from py_rnn.default_params import get_default_params

main_folder = str(Path(__file__).absolute().parent.parent.parent)
data_dir = main_folder + "/data/student_teacher/"  # store inferred model
model_dir = main_folder + "/models/students/"  # store teacher RNN
cuda = True  # toggle if GPU is available
Rz = 0.05
Rx = 0.1
n_repeats = 1
bs = 10
n_epochs = 1500
wandb = False

# Initialise VI / student setup


# train or load teacher RNN
rnn_osc, model_params, task_params, training_params = load_rnn(data_dir + "osc_rnn")
# Extract weights
U, V, B = extract_orth_basis_rnn(rnn_osc)

# initialize task
task_params = {
    "name": "Sine",
    "dur": 75,
    "n_trials": 400,
    "name": "Sine",
    "n_neurons": 20,
    "out": "currents",
    "R_x": Rx,
    "R_z": Rz,
    # "r0":2,
    "non_lin": nn.ReLU(),
}


# initialise encoder
enc_params = {"obs_grad": True, "init_scale": 0.1}
dim_z = 2
dim_N = task_params["n_neurons"]
dim_x = task_params["n_neurons"]
# initialise prior
rnn_params = {
    "train_noise_x": True,
    "train_noise_z": True,
    "train_noise_z_t0": True,
    "init_noise_z": 0.1,
    "init_noise_z_t0": 1,
    "init_noise_x": task_params["R_x"],
    "noise_z": "full",
    "noise_x": "diag",
    "noise_z_t0": "full",
    "identity_readout": True,
    "activation": "relu",
    "decay": 0.7,
    "readout_from": task_params["out"],
    "train_obs_bias": False,
    "train_obs_weights": False,
    "train_neuron_bias": True,
    "weight_dist": "uniform",
    "weight_scaler": 1,  # /dim_N,
    "initial_state": "trainable",
    "out_nonlinearity": "identity",
}


training_params = {
    "lr": 1e-3,
    "lr_end": 1e-5,
    "grad_norm": 0,
    "n_epochs": n_epochs,
    "eval_epochs": 50,
    "batch_size": bs,
    "cuda": cuda,
    "smoothing": 20,
    "freq_cut_off": 10000,
    "k": 64,
    "loss_f": "opt_smc",
    "resample": "systematic",  # , multinomial or none"
    "run_eval": True,
    "smooth_at_eval": False,
}


VAE_params = {
    "dim_x": dim_x,
    "dim_z": dim_z,
    "dim_N": dim_N,
    "enc_architecture": "Inv_Obs",
    "enc_params": enc_params,
    "rnn_architecture": "LRRNN",
    "rnn_params": rnn_params,
}
for i in range(n_repeats):

    task = Oscillations_Cont(task_params, U, V, B)

    vae = VAE(VAE_params)
    # Train
    name = "Noise_test_BS" + str(Rz)
    temp = name.rsplit(".")
    name = temp[0] + temp[1] + datetime.datetime.now().strftime("%Y_%m_%d_T_%H_%M_%S")

    train_VAE(
        vae, training_params, task, sync_wandb=wandb, out_dir=model_dir, fname=name
    )

    print("True noise: " + str(Rz))
    print("Inferred diag noise std:")
    vae = orthogonalise_network(vae)
    print(vae.rnn.std_embed_z(vae.rnn.R_z).detach().numpy())
