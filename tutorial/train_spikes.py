import numpy as np
import torch
import sys, os
from vi_rnn.vae import VAE
sys.path.append("../")

seed = 1
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
from vi_rnn.datasets import Basic_dataset

from vi_rnn.saving import save_model, load_model
from vi_rnn.train import train_VAE
# load data
data_all = np.load("tutorial_data/spiking_data.npy")
dim_x, seq_len = data_all.shape

# split into train and eval
data_train = data_all[:,:75000]
data_eval = data_all[:,75000:]


# initialise a dataset class
task_params = {"name": "tutorial_cont",
               "dur": 100, # we will sample pseudo trials of "dur" timesteps during training
               "n_trials": 256 # every epoch consists of 256 psuedo trials
               }
dataset = Basic_dataset(
    task_params=task_params,
    data=data_train, 
    data_eval=data_eval, 
    stim = None, # you could additionally pass stimuli like this
    stim_eval = None,  
)

enc_params = (
    {"kernel_sizes": [21, 11, 1], # kernel sizes of the CNN
    "padding_mode": "constant", # padding mode of the CNN (e.g., "circular", "constant", "reflect")
    "nonlinearity": "gelu", # "leaky_relu" or "gelu"
    "n_channels": [64,64], # number of channels in the CNN (last one will be equal to dim_z)
    "init_scale": 0.1, # initial scale of the noise predicted by the encoder
    "constant_var": False, # whether or not to use a constant variance (as opposed to a data-dependent variance)
    "padding_location": "acausal",} # padding location of the CNN ("causal", "acausal", or "windowed")
)  
rnn_params = {
    # noise covariances settings
    "train_noise_z": True,  # whether or not to train the transition noise scale
    "train_noise_z_t0": True,  # whether or not to train the initial state noise scale
    "init_noise_z": 0.1,  # initial scale of the transition noise
    "init_noise_z_t0": 0.1,  # initial scale of the initial state noise
    "noise_z": "diag",  # transition noise covariance type ("full", "diag" or "scalar"), set to "full" when using the optimal proposal
    "noise_z_t0": "diag",  # initial state noise covariance type ("full", "diag" or "scalar"), set to "full" when using the optimal proposal
    
    # readout settings
    "identity_readout": True,  # if True enforces a one to one mapping between RNN units and recorded units
    "readout_from": "currents",  # set to "currents", "rates", "z" or "z_and_v". We can readout from the RNN activity
    # before / after applying the non-linearty by setting this to "currents" / "rates" respectively.
    # Alternatively we can directly readout from the latent dynamics z of the RNN by
    # setting this to "z", or from latents z and input v, by setting this to "z_and_v"
    "train_obs_bias": True,  # whether or not to train a bias term in the observation model
    "train_obs_weights": True,  # whether or not train the weights of the observation model
    "out_nonlinearity": "softplus",  # can be used to rectify the output (e.g., when using Poisson observations, use "softplus")
    
    # other
    "activation": "relu",  # set the nonlinearity to "clipped_relu, "relu", "tanh" or "identity"
    "decay": 0.7,  # initial decay constant, scalar between 0 and 1
    "train_neuron_bias": True,  # train a bias term for every neuron
    "weight_dist": "uniform",  # weight distribution ("uniform" or "gauss")
    "initial_state": "trainable",  # initial state ("trainable", "zero", or "bias")
}


VAE_params = {
    "dim_x": 40,  # observation dimension (number of units in the data)
    "dim_z": 2,  # latent dimension / rank of the RNN
    "dim_N": 40,  # amount of units in the RNN (can generally be different then the observation dim)
    "dim_u": 0,  # input stimulus dimension
    "enc_architecture": "CNN",  # encoder architecture (not trained when using linear Gauss observations)
    "enc_params": enc_params,  # encoder paramaters
    "rnn_architecture": "LRRNN",  # use a low-rank RNN architecture
    "rnn_params": rnn_params,  # parameters of the RNN
}

training_params = {
    "lr": 1e-3,  # learning rate start
    "lr_end": 1e-5,  # learning rate end (with exponential decay)
    "n_epochs": 1500,  # number of epochs to train
    "grad_norm": 0,  # gradient clipping above certain norm (if this is set to >0)
    "batch_size": 16,  # batch size
    "cuda": False,  # train on GPU
    "k": 64,  # number of particles to use
    "loss_f": "smc",  # use regular variational SMC ("smc"), or use the optimal ("opt_smc")
    "resample": "systematic",  # type of resampling "systematic", "multinomial" or "none"
    "run_eval": False,  # run an evaluation setup during training (requires additional parameters)
    "observation_likelihood": "Poisson",  # "Gauss"-ian  or "Poisson" observations
    "sim_v": False,  # set to True when using time-varying inputs
    "t_forward":0, # timesteps to predict without using the encoder
}

# initialise the VAE
vae = VAE(VAE_params)

# run training
train_VAE(
    vae,
    training_params,
    dataset,
    sync_wandb=True,
    out_dir="tutorial_data",
    fname="tutorial_spikes",
)