import numpy as np
import sys, os

vi_rnn_dir = os.path.dirname(os.path.abspath(__file__)) + "/../.."
sys.path.append(vi_rnn_dir)
from vi_rnn.vae import VAE
from vi_rnn.train import train_VAE
from vi_rnn.datasets import Basic_dataset


# We used openly accessible electroencephalogram (EEG) data from Schalk et al. 2004
# available from https://www.physionet.org/content/eegmmidb/1.0.0/ (Moody et al. 2000; ODC-BY licence).
# This repo includes preprocessed data from session S001R01

# Set key parameters
# ------------------
dim_z = 3  # latent dimensionality
dim_N = 512  # number of neurons
n_runs = 1  # number of runs
data_eval_name = "EEG_data_smoothed.npy"  # Use smooth on data
data_name = "EEG_data_zscored.npy"  # Use raw (but zcored) data for training
wandb = True  # Sync with wandb
n_epochs = 1500  # number of epochs
bs = 10  # batch size
cuda = True  # use cuda
out_dir = vi_rnn_dir + "/models/sweep_eeg/"  # output directory
data_path = vi_rnn_dir + "/data/eeg/"  # data directory

if str(os.popen("hostname").read()) == "Matthijss-MacBook-Air\n":
    cuda = False
elif str(os.popen("hostname").read()) == "MatthijsDesktop\n":
    cuda = True
else:
    out_dir = "/mnt/qb/work/macke/mpals85/vi_rnn_models/sweepJul27/"
    data_path = "/home/macke/mpals85/vi_rnns/data/eeg/"
    cuda = True


# initialise dataset
# ------------------
task_params = {"name": "EEG", "dur": 50, "n_trials": 50 * bs}
data = np.float32(np.load(data_path + data_name)).T
data_eval = np.float32(np.load(data_path + data_eval_name)).T
task = Basic_dataset(task_params, data, data_eval)
dim_x = task.data.shape[0]


# Train the VAE
# ------------------
for _ in range(n_runs):
    # initialise encoder
    enc_params = {}

    # initialise prior
    rnn_params = {
        "train_noise_x": True,
        "train_noise_z": True,
        "train_noise_z_t0": True,
        "init_noise_z": 0.1,
        "init_noise_z_t0": 1,
        "init_noise_x": 0.1,
        "noise_z": "full",
        "noise_x": "diag",
        "noise_z_t0": "full",
        "identity_readout": False,
        "activation": "clipped_relu",
        "decay": 0.9,
        "readout_from": "z",
        "train_obs_bias": True,
        "train_obs_weights": True,
        "train_neuron_bias": True,
        "weight_dist": "uniform",
        "initial_state": "trainable",
        "out_nonlinearity": "identity",
    }

    training_params = {
        "lr": 1e-3,
        "lr_end": 1e-6,
        "n_epochs": n_epochs,
        "grad_norm": 0,
        "eval_epochs": 25,
        "batch_size": bs,
        "cuda": cuda,
        "smoothing": 20,
        "freq_cut_off": -1,
        "k": 10,
        "resample": "systematic",
        "loss_f": "opt_smc",
        "run_eval": True,
        "smooth_at_eval": True,
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

    vae = VAE(VAE_params)

    train_VAE(vae, training_params, task, sync_wandb=wandb, out_dir=out_dir, fname=None)
