import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import time
import os

os.environ["WANDB__SERVICE_WAIT"] = "1000"
import sys
from evaluation.eval_kl_pse import eval_kl_pse
from vi_rnn.generate import generate
from vi_rnn.saving import save_model
from vi_rnn.inference import (
    filtering_posterior_optimal_proposal,
    filtering_posterior,
    filtering_posterior_bootstrap,
)

file_dir = str(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(file_dir + "/..")
sys.path.append(file_dir)

import matplotlib.pyplot as plt

try:
    import wandb
except:
    print("wandb not installed... continuing")


def train_VAE(
    vae,
    training_params,
    task,
    sync_wandb=False,
    out_dir=None,
    fname=None,
    optimizer=None,
    scheduler=None,
    curr_epoch=0,
    store_train_stats=True,
):
    """
    Train an VAE

    Args:
        vae: initialized VAE
        training_params: dictionary of training parameters
        task, Pytorch Dataset
        syn_wandb: Bool, indicates synchronsation with WandB
        out_dir: string designating where to store model
        fname: model name
        optimizer: torch optimizer object (for restarting training)
        scheduler: torch scheduler object (for restarting training)
        curr_epoch: int, epoch to start from (for restarting training)
        store_train_stats: Bool, store training statistics
    """
    stop_training = False  # not found any NANs yet

    # add losses to training_params dict (bit of a hack)
    training_loss_keys = [
        "ll",
        "KL_x",
        "PSH",
        "mean_error",
        "alphan",
    ]
    for key in training_loss_keys:
        if key not in training_params.keys():
            training_params[key] = []

    # cuda management, gpu potentially speeds up training
    if training_params["cuda"]:
        if not torch.cuda.is_available():
            print("Warning: CUDA not available on this machine, switching to CPU")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    vae.to_device(device)
    print("Training on : " + str(device))

    # set up dataloader
    dataloader = DataLoader(
        task, batch_size=training_params["batch_size"], shuffle=True
    )
    dataloader.dataset.data = dataloader.dataset.data.to(device=device)
    dataloader.dataset.data_eval = dataloader.dataset.data_eval.to(device=device)
    dataloader.dataset.stim = dataloader.dataset.stim.to(device=device)
    dataloader.dataset.stim_eval = dataloader.dataset.stim_eval.to(device=device)

    # initialize wandb
    if sync_wandb:
        wandb.init(
            project="vi_rnns",
            group=task.task_params["name"],
            config={**vae.vae_params, **task.task_params, **training_params},
        )
        wandb.watch(vae, log="all")

    # set exponential decay learning rate scheduler with RAdam optimizer
    optimizer = optimizer or torch.optim.RAdam(
        vae.parameters(), lr=training_params["lr"]
    )
    gamma = np.exp(
        np.log(training_params["lr_end"] / training_params["lr"])
        / training_params["n_epochs"]
    )
    print("Learning rate decay factor " + str(gamma))
    scheduler = scheduler or torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma, last_epoch=-1
    )

    # loss function
    losses = []

    # start timer before training
    time0 = time.time()

    # Initialise Proposal
    if training_params["loss_f"] == "opt_smc":
        inference_f = filtering_posterior_optimal_proposal
    elif training_params["loss_f"] == "smc":
        inference_f = filtering_posterior
    elif training_params["loss_f"] == "bs_smc":
        inference_f = filtering_posterior_bootstrap
    else:
        ValueError(
            "proposal not recognised, use opt_proposal, enc_proposal_diag or bootstrap_proposal"
        )

    for i in range(curr_epoch, training_params["n_epochs"]):

        # run evaluation every eval_epochs:
        # ------------------------------
        if training_params["run_eval"] and i % training_params["eval_epochs"] == 0:
            with torch.no_grad():
                vae.eval()

                # calculate some statistics of generated data
                klx_bin, psH, mean_rate_error = eval_kl_pse(
                    vae,
                    task,
                    cut_off=0,
                    init_state_eval=training_params["init_state_eval"],
                    smoothing=training_params["smoothing"],
                    freq_cut_off=training_params["freq_cut_off"],
                    smooth_at_eval=training_params["smooth_at_eval"],
                )
                training_params["KL_x"].append(klx_bin)
                training_params["PSH"].append(psH)
                training_params["mean_error"].append(mean_rate_error)

                # sync to wandb
                if sync_wandb:
                    wandb.log(
                        {
                            "KL_data": klx_bin,
                            "power_spectr_distance": psH,
                            "mean_rate_error": mean_rate_error,
                        }
                    )

                    # plot latent time series and reconstructions
                    data = task.data_eval
                    u = task.stim_eval
                    if len(task.data_eval.shape) == 2:
                        data = data.unsqueeze(0)
                        u = u.unsqueeze(0)
                    dur = min(data.shape[2], 1000)
                    Z, data_gen, _ = generate(
                        vae,
                        u=u,
                        x=data,
                        dur=dur,
                        initial_state=training_params["init_state_eval"],
                        cut_off=0,
                    )
                    plt.figure()
                    plt.plot(Z[0].T)
                    plt.xlim(0)
                    wandb.log({"latent" + str(i): plt})
                    plt.figure()
                    plt.plot(data_gen[0].T)
                    wandb.log({"reconstruction" + str(i): plt})

        # training
        # --------------------
        vae.train()
        batch_ll = 0

        for data, stim in dataloader:

            optimizer.zero_grad()

            Loss_it, _, alphas = inference_f(
                vae, data, stim, training_params["k"], training_params["resample"]
            )

            batch_ll += Loss_it.mean().item()
            loss = -Loss_it.mean()

            # check for nans
            if torch.isnan(loss):
                print("UH OH FOUND NAN, stopping training...")
                stop_training = True
                break

            # backprop
            loss.backward()

            # gradient clipping
            if training_params["grad_norm"]:
                nn.utils.clip_grad_norm_(
                    parameters=vae.parameters(), max_norm=training_params["grad_norm"]
                )

            # update parameters
            optimizer.step()

        if stop_training:
            break

        # compute average loss
        batch_ll /= len(dataloader)

        alpha = torch.mean(alphas).item()

        if store_train_stats:
            training_params["ll"].append(batch_ll)
            training_params["alphan"].append(alpha)

        print(
            "epoch {}  ll: {:.4f},alpha: {:.2f}, lr: {:.6f}".format(
                i + 1,
                batch_ll,
                alpha,
                scheduler.get_last_lr()[0],
            )
        )

        if sync_wandb:
            wandb.log(
                {
                    "ll": batch_ll,
                    "alpha": alpha,
                    "lr": scheduler.get_last_lr()[0],
                }
            )

        if scheduler.get_last_lr()[0] > training_params["lr_end"]:
            scheduler.step()

    print("\nDone. Training took %.1f sec." % (time.time() - time0))

    # save trained network
    fname = save_model(
        vae, training_params, task.task_params, directory=out_dir, name=fname
    )
    print("Saved: " + fname)
    # upload trained models to WandB
    if sync_wandb:
        # store to wandb
        print(fname + "_state_dict_enc.pkl")
        if vae.has_encoder:
            wandb.save(fname + "_state_dict_enc.pkl")
        wandb.save(fname + "_state_dict_rnn.pkl")
        wandb.save(fname + "_vae_params.pkl")
        wandb.save(fname + "_task_params.pkl")
        wandb.save(fname + "_training_params.pkl")
        wandb.finish()

    return losses
