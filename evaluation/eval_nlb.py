import sys
from pathlib import Path

sys.path.insert(1, str(Path(__file__).absolute().parent.parent.parent))
sys.path.insert(1, str(Path(__file__).absolute().parent.parent))

from vi_rnn.saving import load_model
from vi_rnn.datasets import NLBDataset, load_nlb_dataset
from nlb_tools.evaluation import evaluate, fit_and_eval_decoder
import torch
import numpy as np
import pandas as pd
import h5py
import itertools

from omegaconf import OmegaConf


def eval_nlb(
    config,
    vae,
    train_dataset,
    eval_dataset,
    target=None,
    n_particles=100,
    n_repeats=10,
    device="cuda",
    smooth=False,
):
    """Function for the NLB evaluation
    Args:
        config (OmegaConf): configuration file
        vae (VAE): trained VAE model
        train_dataset (NLBDataset): training
        eval_dataset (NLBDataset): evaluation dataset
        target (str): test_annotation_file
        n_particles (int): number of particles
        n_repeats (int): number of repeats (over which we average)
        device (str): device to use
        smooth (bool): whether to smooth (instead of filter)
    Returns:
        results (dict): evaluation results
        submission (dict): submission dictionary

    """
    t_held_in = eval_dataset.data.shape[2]
    n_held_in = eval_dataset.data.shape[1]
    t_held_out = train_dataset.data.shape[2] - eval_dataset.data.shape[2]
    name = config.dataset.name.replace("_input", "")
    name = name if config.dataset.bin_size == 5 else f"{name}_{config.dataset.bin_size}"

    x_t = train_dataset.data.to(device)
    x_e = eval_dataset.data.to(device)
    u_t = train_dataset.stim.to(device)
    u_e = eval_dataset.stim.to(device)

    eval_predictions = []
    training_predictions = []
    return_idx = 3 if smooth else 2
    for i in range(n_repeats):
        with torch.no_grad():
            Xs_t = np.concatenate(
                [
                    vae.predict_NLB(
                        x_t_chunk,
                        u=u_t_chunk,
                        k=n_particles,
                        t_held_in=t_held_in,
                        t_forward=t_held_out,
                    )[return_idx]
                    .detach()
                    .cpu()
                    .numpy()
                    for x_t_chunk, u_t_chunk in zip(
                        torch.chunk(
                            x_t,
                            chunks=len(x_t) // config.training_params.batch_size // 2,
                            dim=0,
                        ),
                        torch.chunk(
                            u_t,
                            chunks=len(u_t) // config.training_params.batch_size // 2,
                            dim=0,
                        ),
                    )
                ]
            )
            Xs_e = np.concatenate(
                [
                    vae.predict_NLB(
                        x_e_chunk,
                        u=u_e_chunk,
                        k=n_particles,
                        t_held_in=t_held_in,
                        t_forward=t_held_out,
                    )[return_idx]
                    .detach()
                    .cpu()
                    .numpy()
                    for x_e_chunk, u_e_chunk in zip(
                        torch.chunk(
                            x_e,
                            chunks=len(x_e) // config.training_params.batch_size // 2,
                            dim=0,
                        ),
                        torch.chunk(
                            u_e,
                            chunks=len(u_e) // config.training_params.batch_size // 2,
                            dim=0,
                        ),
                    )
                ]
            )

            training_predictions_i = Xs_t.mean(axis=-1).transpose(0, 2, 1)
            training_predictions.append(training_predictions_i)
            eval_predictions_i = Xs_e.mean(axis=-1).transpose(0, 2, 1)
            eval_predictions.append(eval_predictions_i)

    eval_predictions = np.array(eval_predictions).mean(axis=0)
    training_predictions = np.array(training_predictions).mean(axis=0)

    submission = {
        name: {
            "train_rates_heldin": training_predictions[:, :t_held_in, :n_held_in],
            "train_rates_heldout": training_predictions[:, :t_held_in, n_held_in:],
            "eval_rates_heldin": eval_predictions[:, :t_held_in, :n_held_in],
            "eval_rates_heldout": eval_predictions[:, :t_held_in, n_held_in:],
        }
    }
    if t_held_out > 0:
        submission[name]["train_rates_heldin_forward"] = training_predictions[
            :, t_held_in:, :n_held_in
        ]
        submission[name]["train_rates_heldout_forward"] = training_predictions[
            :, t_held_in:, n_held_in:
        ]
        submission[name]["eval_rates_heldin_forward"] = eval_predictions[
            :, t_held_in:, :n_held_in
        ]
        submission[name]["eval_rates_heldout_forward"] = eval_predictions[
            :, t_held_in:, n_held_in:
        ]
    if target is not None:
        results = evaluate(str(target), submission)[0]
        results = results[list(results.keys())[0]]
    else:
        results = None

    return results, submission


def eval_velocity(
    config,
    vae,
    train_dataset,
    target,
    n_particles=100,
    n_repeats=10,
    frac_train=0.7,
    device="cuda",
    smooth=True,
):
    """
    Function for velocity decoding
    Args:
        config (OmegaConf): configuration file
        vae (VAE): trained VAE model
        train_dataset (NLBDataset): training dataset
        target (str): target file
        n_particles (int): number of particles
        n_repeats (int): number of repeats
        frac_train (float): fraction of training data
        device (str): device to use
        smooth (bool): whether to smooth (instead of filter)
    Returns:
        r2 (float): R2 score
        (training_predictions, eval_predictions) (tuple): predictions

    """
    name = (
        "mc_maze"
        if config.dataset.name
        in ["mc_maze_input", "mc_maze_input_long", "mc_maze_input_long_v2"]
        else config.dataset.name
    )
    name = name if config.dataset.bin_size == 5 else f"{name}_{config.dataset.bin_size}"

    n_train = int(len(train_dataset.data) * frac_train)
    x_t = train_dataset.data.to(device)
    x_e = x_t[n_train:]
    x_t = x_t[:n_train]
    u_t = train_dataset.stim.to(device)
    u_e = u_t[n_train:]
    u_t = u_t[:n_train]
    t_held_in = x_t.shape[-1]
    t_held_out = 0

    eval_predictions = []
    training_predictions = []
    return_idx = 3 if smooth else 2
    for i in range(n_repeats):
        with torch.no_grad():
            Xs_t = np.concatenate(
                [
                    vae.predict_NLB(
                        x_t_chunk,
                        u=u_t_chunk,
                        k=n_particles,
                        t_held_in=t_held_in,
                        t_forward=t_held_out,
                    )[return_idx]
                    .detach()
                    .cpu()
                    .numpy()
                    for x_t_chunk, u_t_chunk in zip(
                        torch.chunk(
                            x_t,
                            chunks=len(x_t) // config.training_params.batch_size // 2,
                            dim=0,
                        ),
                        torch.chunk(
                            u_t,
                            chunks=len(u_t) // config.training_params.batch_size // 2,
                            dim=0,
                        ),
                    )
                ]
            )
            Xs_e = np.concatenate(
                [
                    vae.predict_NLB(
                        x_e_chunk,
                        u=u_e_chunk,
                        k=n_particles,
                        t_held_in=t_held_in,
                        t_forward=t_held_out,
                    )[return_idx]
                    .detach()
                    .cpu()
                    .numpy()
                    for x_e_chunk, u_e_chunk in zip(
                        torch.chunk(
                            x_e,
                            chunks=len(x_e) // config.training_params.batch_size // 2,
                            dim=0,
                        ),
                        torch.chunk(
                            u_e,
                            chunks=len(u_e) // config.training_params.batch_size // 2,
                            dim=0,
                        ),
                    )
                ]
            )

            training_predictions_i = Xs_t.mean(axis=-1).transpose(0, 2, 1)
            training_predictions.append(training_predictions_i)
            eval_predictions_i = Xs_e.mean(axis=-1).transpose(0, 2, 1)
            eval_predictions.append(eval_predictions_i)

    eval_predictions = np.array(eval_predictions).mean(axis=0)
    training_predictions = np.array(training_predictions).mean(axis=0)

    with h5py.File(target, "r") as h5f:
        train_behavior = h5f[name]["train_behavior"][()]
    training_behavior = train_behavior[:n_train]
    eval_behavior = train_behavior[n_train:]
    flatten = lambda x: x.reshape(-1, x.shape[-1])
    r2 = fit_and_eval_decoder(
        flatten(training_predictions),
        flatten(training_behavior),
        flatten(eval_predictions),
        flatten(eval_behavior),
        grid_search=True,
    )
    return r2, (training_predictions, eval_predictions)


if __name__ == "__main__":
    # configure evaluation
    DATA_ROOT = Path(__file__).absolute().parent.parent / "data_untracked" / "processed"
    RUN_ROOT = Path(__file__).absolute().parent.parent.parent.parent / "runs"
    # /home/matthijs/runs/reach_nlb/
    RUN_NAME = "maze_rs1"
    MODEL_NAME = "004"
    CHKPT = "best"
    PHASE = "test"
    SWEEP = False
    smooth = False
    n_particles_sweep = [64, 128, 192]
    n_repeats_sweep = [16, 32]

    model_dir = RUN_ROOT / RUN_NAME / MODEL_NAME
    model_dir = Path("/home/matthijs/runs/reach_nlb/")

    config = OmegaConf.load(model_dir / "config.yaml")
    chkpt_path = model_dir / "checkpoints" / CHKPT
    model_save_name = sorted(chkpt_path.glob("*.pkl"))[0].stem

    # load model
    suffixes = [
        "_state_dict_enc",
        "_state_dict_prior",
        "_task_params",
        "_training_params",
        "_vae_params",
    ]
    for suffix in suffixes:
        model_save_name = model_save_name.replace(suffix, "")
    model_save_name = str(chkpt_path / model_save_name)
    vae, vae_params, task_params, training_params = load_model(model_save_name)
    # set up dataset
    dataset_config = {**config.dataset}
    dataset_config["phase"] = "val"
    train_data, eval_data, train_inputs, eval_inputs = load_nlb_dataset(
        data_root=DATA_ROOT, **dataset_config
    )
    train_dataset = NLBDataset(
        train_data, dict(name=config.dataset.name), inputs=train_inputs
    )
    eval_dataset = NLBDataset(
        eval_data, dict(name=config.dataset.name), inputs=eval_inputs
    )

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

    target_path = (
        DATA_ROOT
        / dataset_config["name"]
        / dataset_config["phase"]
        / f"eval_target_{dataset_config['bin_size']}ms.h5"
    )
    target = target_path if target_path.exists() else None

    if SWEEP:
        result_df = []
        for n_particles, n_repeats in itertools.product(
            n_particles_sweep, n_repeats_sweep
        ):
            print(f"{n_particles=}, {n_repeats=}")
            results = eval_nlb(
                config,
                vae,
                train_dataset,
                eval_dataset,
                target=target,
                n_particles=n_particles,
                n_repeats=n_repeats,
                device="cuda",
                smooth=smooth,
            )[0]
            result_df.append(
                dict(n_particles=n_particles, n_repeats=n_repeats, **results)
            )
            print(results)
        result_df = pd.DataFrame(result_df)
        result_df.to_csv("sweep.csv")
        best_index = result_df["co-bps"].argmin()
        n_particles = int(round(result_df.iloc[best_index].n_particles))
        n_repeats = int(round(result_df.iloc[best_index].n_repeats))
    else:
        n_particles = 192
        n_repeats = 32

    dataset_config = {**config.dataset}
    dataset_config["phase"] = PHASE
    print(dataset_config)
    train_data, eval_data, train_inputs, eval_inputs = load_nlb_dataset(
        data_root=DATA_ROOT, **dataset_config
    )
    train_dataset = NLBDataset(
        train_data, dict(name=config.dataset.name), inputs=train_inputs
    )
    eval_dataset = NLBDataset(
        eval_data, dict(name=config.dataset.name), inputs=eval_inputs
    )

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

    target_path = (
        DATA_ROOT
        / dataset_config["name"]
        / dataset_config["phase"]
        / f"eval_target_{dataset_config['bin_size']}ms.h5"
    )
    target = target_path if target_path.exists() else None

    results, submission = eval_nlb(
        config,
        vae,
        train_dataset,
        eval_dataset,
        target=target,
        n_particles=n_particles,
        n_repeats=n_repeats,
        device="cuda",
        smooth=smooth,
    )
    print(results)

    from nlb_tools.make_tensors import save_to_h5

    save_to_h5(
        submission,
        "_".join(
            [
                dataset_config["name"],
                dataset_config["phase"],
                str(dataset_config["bin_size"]),
                "submission.h5",
            ]
        ),
    )
