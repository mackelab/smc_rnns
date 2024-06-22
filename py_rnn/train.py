import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pickle
import time
import os
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
from model import *

try:
    import wandb
except:
    print("wandb not installed... continuing")


def train_rnn(
    rnn, training_params, task, sync_wandb=False, wandb_log_freq=100, x0=None
):
    """
    Train a biologically inspired RNN

    Args:
        rnn: initialized RNN
        training_params: dictionary of training parameters
        task, Pytorch Dataset should on call return:
                            trial, of size [seq_len, n_inp]
                            target, of size [seq_len, n_out]
                            mask, of size [seq_len, n_out]
        syn_wandb (optional): Bool, indicates synchronsation with WandB
        wandb_log_freq: Int, how often to synchronise gradients + weights
    """

    dataloader = DataLoader(
        task, batch_size=training_params["batch_size"], shuffle=True
    )

    # cuda management, gpu highly speeds up training
    if training_params["cuda"]:
        if not torch.cuda.is_available():
            print("Warning: CUDA not available on this machine, switching to CPU")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    rnn.to(device=device)

    # choose a loss function
    if training_params["loss_fn"] == "mse":
        loss_fn = mse_loss
    elif training_params["loss_fn"] == "cos":
        loss_fn = cos_loss
    elif training_params["loss_fn"] == "none":
        loss_fn = zero_loss
    else:
        print("WARNING: Loss function not implemented")

    reg_fns = []
    reg_costs = []
    # regulisation
    if training_params["osc_reg_cost"]:
        reg_fns.append(
            LFPLoss(
                freq=training_params["osc_reg_freq"],
                tstep=rnn.params["dt"] / 1000,
                T=dataloader.dataset[0][0].size(0),
                device=device,
            )
        )
        reg_costs.append(training_params["osc_reg_cost"])
    if training_params["offset_reg_cost"]:
        reg_fns.append(offset_loss)
        reg_costs.append(training_params["offset_reg_cost"])
    if training_params["l2_rates_reg"]:
        print("using l2 rates reg")
        reg_fns.append(l2_rates_reg)
        reg_costs.append(training_params["l2_rates_reg"])
    if len(reg_fns) == 0:
        reg_fns.append(zero_loss)
        reg_costs.append(0)
    reg_costs = torch.tensor(reg_costs, device=device)
    # optimizer
    if training_params["optimizer"] == "adam":
        optimizer = torch.optim.Adam(rnn.parameters(), lr=training_params["lr"])

    # initialize wandb
    if sync_wandb:
        wandb.init(
            project="phase-coding",
            group="pytorch",
            config={**rnn.params, **dataloader.dataset.task_params, **training_params},
        )
        config = wandb.config
        wandb.watch(rnn, log="all", log_freq=wandb_log_freq)

    # start timer before training
    time0 = time.time()
    # set rnn to training mode
    rnn.train()

    losses = []
    reg_losses = []

    # set to some ridiculous value if not specified, so it doesn't get used
    if training_params["clip_gradient"] is None:
        training_params["clip_gradient"] = 100000

    # start training loop
    for i in range(training_params["n_epochs"]):
        loss_ep = 0.0
        reg_loss_ep = torch.zeros(len(reg_fns), device=device)
        num_len = 0

        # loop through dataloader
        for x, y, m in dataloader:

            x = x.to(device=device)
            y = y.to(device=device)
            m = m.to(device=device)

            rates, y_pred = rnn(x, x0)
            optimizer.zero_grad()
            task_loss = loss_fn(y_pred, y, m)
            reg_loss = torch.stack(
                [reg_fn(rates, rnn.rnn) for reg_fn in reg_fns]
            ).squeeze()  # , device=device)
            # grad descent
            loss = task_loss + torch.sum(reg_loss * reg_costs)
            loss.backward()

            # clip weights to avoid explosion

            torch.nn.utils.clip_grad_norm_(
                rnn.parameters(), training_params["clip_gradient"]
            )

            # update weights
            optimizer.step()
            num_len += 1
            loss_ep += task_loss.item()
            reg_loss_ep += reg_loss  # .tolist()

        # print average loss and print / sync
        loss_ep /= num_len
        reg_loss_ep /= num_len
        reg_loss_ep = reg_loss_ep.tolist()
        print(
            "epoch {:d} / {:d}: time={:.1f} s, task loss={:.5f}, reg loss=".format(
                i + 1, training_params["n_epochs"], time.time() - time0, loss_ep
            )
            + str(["{:.5f}"] * len(reg_loss_ep))
            .format(*reg_loss_ep)
            .strip("[]")
            .replace("'", ""),
            end="\r",
        )
        if sync_wandb:
            wandb.log({"task_loss": loss_ep, "reg_los": reg_loss_ep})
        losses.append(loss_ep)
        reg_losses.append(reg_loss_ep)
    print("\nDone. Training took %.1f sec." % (time.time() - time0))
    if sync_wandb:
        wandb.finish()
    rnn.eval()
    return losses, reg_losses


def load_rnn(name):
    """
    loads an RNN

    Args:
        name: String, path / name to where RNN is saved

    Returns:
        model: Initialized RNN
        params: dictionary of model parameters
        task_params: dictionary of task parameters
        training_params: dictionary of training parameters
    """

    state_dict_file = name + "_state_dict.pkl"
    params_file = name + "_params.pkl"
    task_params_file = name + "_task_params.pkl"
    training_params_file = name + "_training_params.pkl"

    with open(params_file, "rb") as f:
        params = pickle.load(f)
    with open(task_params_file, "rb") as f:
        task_params = pickle.load(f)
    with open(training_params_file, "rb") as f:
        training_params = pickle.load(f)

    model = RNN(params)
    model.load_state_dict(torch.load(state_dict_file))

    return model, params, task_params, training_params


def save_rnn(name, model, params, task_params, training_params):
    """
    saves an RNN

    Args:
        name: String, path / name to where RNN is saved
        model: Initialized RNN
        params: dictionary of model parameters
        task_params: dictionary of task parameters
        training_params: dictionary of training parameters
    """

    state_dict_file = name + "_state_dict.pkl"
    params_file = name + "_params.pkl"
    task_params_file = name + "_task_params.pkl"
    training_params_file = name + "_training_params.pkl"
    with open(params_file, "wb") as f:
        pickle.dump(params, f)
    with open(training_params_file, "wb") as f:
        pickle.dump(training_params, f)
    with open(task_params_file, "wb") as f:
        pickle.dump(task_params, f)

    torch.save(model.state_dict(), state_dict_file)


def extract_lfp(x, rnn_cell, normalize=True):
    """
    Calculate LFP as mean absolute synaptic input

    Args:
        x: currents throughout trials, Tensor of size [batch_size, seq_len, n_rec]
        rnn_cell: calculates forward pass of an RNN
        normalize(optional): zscore LFP

    Returns:
       lfp: local field potential, Tensor of size [batch_size, seq_len]
    """

    w_eff = rnn_cell.dale_mask(rnn_cell.w_rec)
    w_eff = rnn_cell.conn_mask(w_eff)
    tau = rnn_cell.project_taus(rnn_cell.taus_gaus, rnn_cell.tau)
    alpha = rnn_cell.dt / tau

    # mean absolute synaptic input
    abs_inp = alpha * torch.matmul(rnn_cell.nonlinearity(x), torch.abs(w_eff.t()))
    lfp = torch.mean(abs_inp, dim=-1)

    if normalize:
        mean = torch.mean(lfp, dim=1).unsqueeze(1)
        var = torch.mean((lfp - mean.detach()) ** 2, dim=1).unsqueeze(1)
        lfp = (lfp - mean) / torch.sqrt(2 * var)

    return lfp


def l2_rates_reg(rates, *args):
    """l2 reg on non zero mean single unit firing rates"""
    return torch.mean(rates**2)


def offset_loss(rates, *args):
    """l2 reg on non zero mean single unit firing rates"""
    return torch.mean(torch.mean(rates[:, 320:], dim=1) ** 2)


class LFPLoss(object):
    def __init__(self, freq, tstep, T, device):
        """
        Regularizer to promote oscillations at specified frequency

        Args:
            freq: target freq in Hz
            tstep: timestep in S
            T: trial length in model steps
            device: cpu / cuda

        """
        trtime = np.arange(0, tstep * T, tstep, dtype=np.float32)[:T]
        sinF = torch.from_numpy(np.sin(freq * 2 * np.pi * trtime))
        cosF = torch.from_numpy(np.cos(freq * 2 * np.pi * trtime))
        self.sinF = sinF.to(device=device)
        self.cosF = cosF.to(device=device)
        self.T = T

    def __call__(self, x, rnn_cell):
        """
        Calculate loss as norm of fourier component

        Args:
            x: currents, Tensor of size [batch_size, seq_len, n_rec]
            rnn_cell: to calculate a forward pass
        """

        lfp = extract_lfp(x, rnn_cell)
        a = torch.tensordot(self.sinF, lfp, dims=[[0], [1]]) / self.T
        b = torch.tensordot(self.cosF, lfp, dims=[[0], [1]]) / self.T
        norm = torch.sqrt(a**2 + b**2)
        lfp_loss = 0.5 - torch.mean(norm)
        return lfp_loss


def zero_loss(x, *args):
    """
    Utility function returning zero
    Args:
        x: some tensor with correct device

    Returns:
        0

    """
    return torch.zeros(1, device=x.device)


def mse_loss(output, target, mask):
    """
    Mean squared error loss

    Args:
        output (RNN prediction), Tensor size [batch_size, seq_len, n_out]
        target, Tensor size [batch_size, seq_len, n_out]
        mask, Tensor size [batch_size, seq_len, n_out]

    Returns:
        loss

    """
    loss = (mask * (target - output).pow(2)).sum() / mask.sum()
    return loss


def cos_loss(output, target, mask):
    """
    Loss based on vector angle (needs n_out>=2)

    Args:
        output (RNN prediction), Tensor size [batch_size, seq_len, n_out]
        target, Tensor size [batch_size, seq_len, n_out]
        mask, Tensor size [batch_size, seq_len, n_out]

    Returns:
        loss

    """
    criterion = nn.CosineSimilarity(dim=2)
    loss = 0.5 - 0.5 * ((mask.squeeze() * criterion(output, target)).sum() / mask.sum())
    return loss
