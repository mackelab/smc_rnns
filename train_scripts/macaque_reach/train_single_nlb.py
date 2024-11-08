import sys, os
import shutil
from pathlib import Path

sys.path.insert(1, str(Path(__file__).absolute().parent.parent.parent))

from vi_rnn.vae import VAE
from vi_rnn.train import train_VAE
from vi_rnn.datasets import NLBDataset, load_nlb_dataset
from evaluation.eval_nlb import eval_nlb
import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py

import hydra
from omegaconf import OmegaConf

DATA_ROOT = (
    Path(__file__).absolute().parent.parent.parent / "data_untracked" / "processed"
)
RUN_ROOT = Path(__file__).absolute().parent.parent.parent.parent / "runs"


def train(
    run_name,
    overrides: dict = {},
    config_path: str = "configs/single.yaml",
    evaluate: bool = True,
    plot: bool = True,
    patience: int = 50,
):
    config_path = Path(config_path)
    overrides = [f"{k}={v}" for k, v in overrides.items()]
    with hydra.initialize(
        config_path=str(config_path.parent),
        job_name="train_vae",
        version_base="1.1",
    ):
        config = hydra.compose(config_name=config_path.name, overrides=overrides)

    vae = VAE(OmegaConf.to_object(config.vae_params))

    train_data, eval_data, train_inputs, eval_inputs = load_nlb_dataset(
        data_root=DATA_ROOT, **OmegaConf.to_object(config.dataset)
    )
    train_dataset = NLBDataset(
        train_data, dict(name=config.dataset.name), inputs=train_inputs
    )
    eval_dataset = NLBDataset(
        eval_data, dict(name=config.dataset.name), inputs=eval_inputs
    )

    run_dir = RUN_ROOT / run_name
    if not os.path.isdir(run_dir):
        run_dir.mkdir(parents=True)
    chkpt_dir = run_dir / "checkpoints"
    if not os.path.isdir(chkpt_dir / "last"):
        (chkpt_dir / "last").mkdir(parents=True)
    if not os.path.isdir(chkpt_dir / "best"):
        (chkpt_dir / "best").mkdir(parents=True)
    OmegaConf.save(config, run_dir / "config.yaml")

    if config.training_params["cuda"]:
        if not torch.cuda.is_available():
            print("Warning: CUDA not available on this machine, switching to CPU")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    vae.to_device(device)
    train_dataset.data = train_dataset.data.to(device)
    train_dataset.data_eval = train_dataset.data_eval.to(device)
    train_dataset.stim = train_dataset.stim.to(device)
    eval_dataset.data = eval_dataset.data.to(device)
    eval_dataset.data_eval = eval_dataset.data_eval.to(device)
    eval_dataset.stim = eval_dataset.stim.to(device)

    training_params = OmegaConf.to_object(config.training_params)
    n_epochs = training_params["n_epochs"]
    step = training_params["eval_epochs"]
    optimizer = torch.optim.RAdam(vae.parameters(), lr=training_params["lr"])
    gamma = np.exp(
        np.log(training_params["lr_end"] / training_params["lr"])
        / training_params["n_epochs"]
    )
    print("Learning rate decay factor " + str(gamma))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma, last_epoch=-1, verbose=False
    )

    target_path = (
        DATA_ROOT
        / config.dataset.name
        / config.dataset.phase
        / f"eval_target_{config.dataset.bin_size}ms.h5"
    )
    run_eval_nlb = target_path.exists()

    best_loss = np.inf
    nlb_results = []

    if evaluate and run_eval_nlb:
        results, submission = eval_nlb(
            config,
            vae,
            train_dataset,
            eval_dataset,
            target_path,
            device=device,
        )
        print(f"NLB eval: " + ", ".join([f"{k}={v:.4f}" for k, v in results.items()]))

    for i in range(0, n_epochs, step):
        for item in (chkpt_dir / "last").iterdir():
            item.unlink()

        training_params["n_epochs"] = min(i + step, n_epochs)
        out_dir = str(chkpt_dir / "last")
        if not out_dir.endswith("/"):
            out_dir += "/"
        train_VAE(
            vae,
            training_params,
            train_dataset,
            sync_wandb=False,
            out_dir=out_dir,
            fname=None,
            optimizer=optimizer,
            scheduler=scheduler,
            curr_epoch=i,
        )

        if evaluate and run_eval_nlb:
            results, submission = eval_nlb(
                config,
                vae,
                train_dataset,
                eval_dataset,
                target_path,
                device=device,
            )
            print(
                f"NLB eval: " + ", ".join([f"{k}={v:.4f}" for k, v in results.items()])
            )
            nlb_results.append(results)
            if plot:
                train_pred = submission["train_rates_heldin"]
                fig, axs = plt.subplots(1, 3, figsize=(7, 2))
                trial = 0
                axs[0].plot(train_pred[trial, :, 0:5])  # line plot of some rates
                axs[0].set_ylim(0)
                axs[0].set_title("neuron 0-4 rates")
                axs[1].imshow(train_dataset.data.detach().cpu()[trial])  # imshow spikes
                axs[1].set_title("true spikes")
                axs[2].imshow(train_pred[trial])  # imshow rates
                axs[2].set_title("model rates")
                plt.tight_layout()
                plt.savefig(chkpt_dir / "last" / "plots.png", dpi=300)
                plt.close()

        with h5py.File(chkpt_dir / "last" / "metrics.h5", "w") as h5f:
            h5f.create_dataset("loss", data=training_params["loss"][-1])
            if len(nlb_results) > 0:
                for k, v in nlb_results[-1].items():
                    h5f.create_dataset(k, data=v)

        last_loss = -1 * nlb_results[-1]["co-bps"]
        if last_loss < best_loss - 1e-4:
            best_loss = last_loss
            for item in (chkpt_dir / "best").iterdir():
                item.unlink()
            for item in (chkpt_dir / "last").iterdir():
                if item.is_file():
                    shutil.copy(item, chkpt_dir / "best" / item.name)
                elif item.is_dir():
                    shutil.copytree(item, chkpt_dir / "best" / item.name)

        if len(training_params["loss"]) >= patience * 2:
            prev_best_loss = np.min(training_params["loss"][:-patience])
            if not np.any(
                np.array(training_params["loss"][-patience:]) < prev_best_loss
            ):
                print(
                    f"No improvement for the last {patience} iters. Stopping early..."
                )
                break
        if np.any(np.array(training_params["alphan"][-step:]) < 5e-3):
            print(f"Alpha dropped below 0.005. Stopping early...")
            break

    return training_params["loss"], nlb_results


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Train model on NLB dataset")
    parser.add_argument("--run_name", "-r", type=str)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--no_evaluate", action="store_true")
    parser.add_argument("--patience", "-p", type=int, default=50)
    args, extras = parser.parse_known_intermixed_args()

    i = 0
    overrides = {}
    while i < len(extras):
        if "=" in extras[i]:
            k, v = extras[i].strip("'- \n").split("=")
            overrides[k] = v
            i += 1
        else:
            k = extras[i].strip("'- \n")
            v = extras[i + 1].strip("'- \n")
            overrides[k] = v
            i += 2

    train(
        run_name=args.run_name,
        evaluate=(not args.no_evaluate),
        patience=args.patience,
        plot=args.plot,
        overrides=overrides,
    )
