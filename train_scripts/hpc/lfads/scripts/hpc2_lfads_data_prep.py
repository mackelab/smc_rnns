import numpy as np
import h5py
import os
import argparse

vi_rnn_dir = os.path.dirname(os.path.abspath(__file__)) + "/../.."

train_data_path = vi_rnn_dir + "/data_untracked/train_hpc2.npy"
test_data_path = vi_rnn_dir + "/data_untracked/test_hpc2.npy"
save_dir = vi_rnn_dir + "/data_untracked"


# arguments
parser = argparse.ArgumentParser()
parser.add_argument("--trial-len", "-T", type=int, default=100)
args = parser.parse_args()
trial_len = args.trial_len

# load train data and split into trials
train_data = np.load(train_data_path).T

train_data = train_data[: int(train_data.shape[0] // trial_len * trial_len)]
train_data = train_data.reshape(-1, trial_len, train_data.shape[-1])  # b x t x n
indices = np.arange(train_data.shape[0])
valid_split_indices = indices[::5]
train_split_indices = indices[~np.isin(indices, valid_split_indices)]
print(train_data[train_split_indices].shape, train_data[valid_split_indices].shape)

with h5py.File(save_dir + "/train_hpc2_lfads.h5", "w") as h5f:
    h5f.create_dataset("train_encod_data", data=train_data[train_split_indices])
    h5f.create_dataset("train_recon_data", data=train_data[train_split_indices])
    h5f.create_dataset("valid_encod_data", data=train_data[valid_split_indices])
    h5f.create_dataset("valid_recon_data", data=train_data[valid_split_indices])
    h5f.create_dataset("train_idx", data=train_split_indices)
    h5f.create_dataset("valid_idx", data=valid_split_indices)

# load test data and split into trials
test_data = np.load(test_data_path).T

test_data = test_data[: int(test_data.shape[0] // trial_len * trial_len)]
test_data = test_data.reshape(-1, trial_len, test_data.shape[-1])  # b x t x n
indices = np.arange(test_data.shape[0])
valid_split_indices = indices[
    ::5
]  # have to still do a train/valid split for lfads-torch compatibility :(
train_split_indices = indices[~np.isin(indices, valid_split_indices)]
print(test_data[train_split_indices].shape, test_data[valid_split_indices].shape)

with h5py.File(save_dir + "/test_hpc2_lfads.h5", "w") as h5f:
    h5f.create_dataset("train_encod_data", data=test_data[train_split_indices])
    h5f.create_dataset("train_recon_data", data=test_data[train_split_indices])
    h5f.create_dataset("valid_encod_data", data=test_data[valid_split_indices])
    h5f.create_dataset("valid_recon_data", data=test_data[valid_split_indices])
    h5f.create_dataset("train_idx", data=train_split_indices)
    h5f.create_dataset("valid_idx", data=valid_split_indices)
