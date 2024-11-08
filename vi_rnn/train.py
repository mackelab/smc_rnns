import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import time
import os

os.environ["WANDB__SERVICE_WAIT"] = "1000"
import sys
from vi_rnn.evaluation import eval_VAE
from vi_rnn.saving import save_model

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
        "ll_x",
        "ll_z",
        "H",
        "loss",
        "reg_loss",
        "KL_x",
        "PSH",
        "PSC",
        "mean_error",
        "noise_z",
        "noise_x",
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

    # initialize wandb
    if sync_wandb:
        wandb.init(
            project="vi_rnns",
            group=task.task_params["name"],
            config={**vae.vae_params, **task.task_params, **training_params},
        )
        config = wandb.config
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

    # backwards compat
    if "sim_v" not in training_params.keys():
        training_params["sim_v"] = False
    elif training_params["sim_v"]:
        print("Simulating V")

    for i in range(curr_epoch, training_params["n_epochs"]):
        with torch.no_grad():
            # DO EVALUATION
            if i % training_params["eval_epochs"] == 0 and training_params["run_eval"]:
                vae.eval()
                with torch.no_grad():
                    klx_bin, psH, mean_rate_error = eval_VAE(
                        vae,
                        task,
                        cut_off=0,
                        smoothing=training_params["smoothing"],
                        freq_cut_off=training_params["freq_cut_off"],
                        sim_obs_noise=training_params["sim_obs_noise"],
                        sim_latent_noise=training_params["sim_latent_noise"],
                        smooth_at_eval=training_params["smooth_at_eval"],
                    )
                    training_params["KL_x"].append(klx_bin)
                    training_params["PSH"].append(psH)
                    training_params["mean_error"].append(mean_rate_error)

                    if sync_wandb:
                        wandb.log(
                            {
                                "KL_data": klx_bin,
                                "power_spectr_distance": psH,
                                "mean_rate_error": mean_rate_error,
                            }
                        )

                        # plot latent time series and reconstructions
                        with torch.no_grad():
                            data, u = task.__getitem__(0)
                            dim_x, _ = data.shape
                            z_hat, Emean, Esigma, eps_s = vae.encoder(data.unsqueeze(0))
                            z0 = z_hat[:, :, :1].squeeze()
                            Z = vae.rnn.get_latent_time_series(
                                time_steps=1000,
                                z0=z0,
                                noise_scale=training_params["sim_latent_noise"],
                            )
                            data_gen = (
                                vae.rnn.get_observation(
                                    Z, noise_scale=training_params["sim_obs_noise"]
                                )
                                .permute(0, 2, 1, 3)
                                .reshape(1000, dim_x)
                            )
                        plt.figure()
                        plt.plot(Z[0, :, :, 0].detach().cpu().T)
                        plt.xlim(0)
                        wandb.log({"latent" + str(i): plt})
                        plt.figure()
                        plt.plot(data_gen.detach().cpu())
                        wandb.log({"reconstruction" + str(i): plt})

        # set rnn to training mode
        vae.train()

        batch_h_loss = 0
        batch_ll = 0
        batch_ll_z = 0
        batch_ll_x = 0
        batch_loss = 0

        for inputs, stim in dataloader:

            optimizer.zero_grad()
            # forward pass

            # optimal proposal, can be used with linear Gaussian observations
            if training_params["loss_f"] == "opt_VGTF":
                (
                    Loss_it,
                    Z,
                    _,
                    ll_x,
                    ll_z,
                    H,
                    log_likelihood,
                    alphas,
                ) = vae.forward_Optimal_VGTF(
                    inputs,
                    u=stim,
                    k=training_params["k"],
                    resample=training_params["resample"],
                    sim_v=training_params["sim_v"],
                )

            # else we learn a parameterised encoder network
            elif training_params["loss_f"] == "VGTF":
                Loss_it, Z, _, ll_x, ll_z, H, log_likelihood, alphas = vae.forward_VGTF(
                    inputs,
                    u=stim,
                    k=training_params["k"],
                    resample=training_params["resample"],
                    out_likelihood=training_params["observation_likelihood"],
                    t_forward=training_params["t_forward"],
                    sim_v=training_params["sim_v"],
                )

            # don't use an encoder, just sample from RNN (bootstrap proposal)
            elif training_params["loss_f"] == "bs_VGTF":
                (
                    Loss_it,
                    Z,
                    _,
                    ll_x,
                    ll_z,
                    H,
                    log_likelihood,
                    alphas,
                ) = vae.forward_bootstrap_VGTF(
                    inputs,
                    u=stim,
                    k=training_params["k"],
                    resample=training_params["resample"],
                    out_likelihood=training_params["observation_likelihood"],
                    t_forward=training_params["t_forward"],
                    sim_v=training_params["sim_v"],
                )

            # deterministic setting (generalised teacher forcing)
            elif training_params["loss_f"] == "GTF":
                Loss_it, Z, _, ll_x, ll_z, H, log_likelihood, alphas = vae.forward_GTF(
                    inputs,
                    u=stim,
                    alpha=training_params["alpha"],
                    sim_v=training_params["sim_v"],
                )

            batch_ll += log_likelihood.mean().item()
            batch_ll_x += ll_x.mean().item()
            batch_ll_z += ll_z.mean().item()
            batch_h_loss += H.mean().item()
            loss = -Loss_it.mean()
            batch_loss += loss.item()

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

            # Adjust learning rate
            optimizer.step()

        if stop_training:
            break

        # compute average loss
        batch_ll /= len(dataloader)
        batch_ll_z /= -len(dataloader)
        batch_h_loss /= -len(dataloader)
        batch_ll_x /= -len(dataloader)
        batch_loss /= len(dataloader)

        noise_z = vae.rnn.std_embed_z(vae.rnn.R_z).detach()
        noise_x = vae.rnn.std_embed_x(vae.rnn.R_x).detach()
        alpha = torch.mean(alphas).item()

        if store_train_stats:
            training_params["ll_z"].append(batch_ll_z)
            training_params["ll_x"].append(batch_ll_x)
            training_params["ll"].append(batch_ll)
            training_params["H"].append(batch_h_loss)
            training_params["loss"].append(batch_loss)
            training_params["noise_z"].append(noise_z)
            training_params["noise_x"].append(noise_x)
            training_params["alphan"].append(alpha)

        print(
            "epoch {} loss: {:.4f}, ll: {:.4f}, ll_x: {:.4f}, ll_z: {:.4f} H: {:.4f}, alpha: {:.2f}, lr: {:.6f}, N_z: {:.4f}, N_x: {:.4f}".format(
                i + 1,
                batch_loss,
                batch_ll,
                batch_ll_x,
                batch_ll_z,
                batch_h_loss,
                alpha,
                scheduler.get_last_lr()[0],
                noise_z.mean().item(),
                noise_x.mean().item(),
            )
        )

        if sync_wandb:
            wandb.log(
                {
                    "loss": batch_loss,
                    "ll": batch_ll,
                    "likelihood_data": batch_ll_x,
                    "likelihood_latent": batch_ll_z,
                    "entropy": batch_h_loss,
                    "alpha": alpha,
                    "noise_z": noise_z.mean().item(),
                    "noise_x": noise_x.mean().item(),
                    "noise_e": torch.exp(vae.encoder.logvar / 2).mean().item(),
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
        wandb.save(fname + "_state_dict_enc.pkl")
        wandb.save(fname + "_state_dict_prior.pkl")
        wandb.save(fname + "_vae_params.pkl")
        wandb.save(fname + "_task_params.pkl")
        wandb.save(fname + "_training_params.pkl")
        wandb.finish()

    return losses
