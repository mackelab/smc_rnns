import datetime
import pickle
import torch
import os
import io
import sys

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
from vae import VAE


def save_model(model, training_params, task_params, name=None, directory=None):
    """
    Save VAE model
    Args:
        model (nn.Module): VAE model
        training_params (dict): dictionary of training parameters
        task_params (dict): dictionary of task parameters
        name (str): name of the model
        directory (str): directory where the model is saved
    Returns:
        name (str): name of the model
    """
    if not name:
        if not directory:
            directory = "../models/"
        elif directory[-1] != "/":
            directory += "/"

        if "enc_architecture" in model.vae_params:
            enc = model.vae_params["enc_architecture"] + "_"

        else:
            enc = ""
        # Generate a name
        name = (
            task_params["name"]
            + "_"
            + enc
            + model.vae_params["rnn_params"]["transition"]
            + "_"
            + model.vae_params["rnn_params"]["observation"]
            + "_dim_z_"
            + str(model.dim_z)
            + "_date_"
            + datetime.datetime.now().strftime("%Y_%m_%d_T_%H_%M_%S")
        )
        print("Saving model as " + str(name))
    else:
        if not directory:
            directory = ""
        elif directory[-1] != "/":
            directory += "/"

    model_params = model.vae_params
    state_dict_file_prior = directory + name + "_state_dict_rnn.pkl"
    state_dict_file_encoder = directory + name + "_state_dict_enc.pkl"

    vae_params_file = directory + name + "_vae_params.pkl"
    task_params_file = directory + name + "_task_params.pkl"
    training_params_file = directory + name + "_training_params.pkl"
    with open(vae_params_file, "wb") as f:
        pickle.dump(model_params, f)
    with open(training_params_file, "wb") as f:
        pickle.dump(training_params, f)
    with open(task_params_file, "wb") as f:
        pickle.dump(task_params, f)

    torch.save(model.rnn.state_dict(), state_dict_file_prior)
    if model.has_encoder:
        torch.save(model.encoder.state_dict(), state_dict_file_encoder)

    return directory + name


def load_model(name, load_encoder=True, backward_compat=False):
    """
    loads a VAE

    Args:
        name: String, path / name to where RNN is saved

    Returns:
        model: Initialized VAE
        vae_params: dictionary of model parameters
        task_params: dictionary of task parameters
        training_params: dictionary of training parameters
    """

    state_dict_file_rnn = name + "_state_dict_rnn.pkl"
    state_dict_file_encoder = name + "_state_dict_enc.pkl"
    params_file = name + "_vae_params.pkl"
    task_params_file = name + "_task_params.pkl"
    training_params_file = name + "_training_params.pkl"

    with open(params_file, "rb") as f:
        vae_params = CPU_Unpickler(f).load()

    with open(task_params_file, "rb") as f:
        task_params = CPU_Unpickler(f).load()
    with open(training_params_file, "rb") as f:
        training_params = CPU_Unpickler(f).load()

    if backward_compat:
        # Backwards compatibility
        if "prior_params" in vae_params:
            vae_params["rnn_params"] = vae_params.pop("prior_params")

        if (
            "enc_architecture" in vae_params
            and vae_params["enc_architecture"] == "CNN_causal"
        ):
            vae_params["enc_architecture"] = "CNN"
            vae_params["enc_params"]["padding_location"] = "causal"

        if (
            "enc_params" in vae_params
            and "padding_location" not in vae_params["enc_params"]
        ):
            vae_params["enc_params"]["padding_location"] = "causal"

        if "readout_v " in vae_params["rnn_params"]:
            _ = vae_params["rnn_params"].pop("readout_v")
            vae_params["rnn_params"]["readout_from"] = "z_and_v"
        if "train_noise_obs" in vae_params["rnn_params"]:
            vae_params["rnn_params"]["train_noise_x"] = vae_params["rnn_params"].pop(
                "train_noise_obs"
            )
        if "train_noise_prior" in vae_params["rnn_params"]:
            vae_params["rnn_params"]["train_noise_z"] = vae_params["rnn_params"].pop(
                "train_noise_prior"
            )
        if "train_noise_prior_t0" in vae_params["rnn_params"]:
            vae_params["rnn_params"]["train_noise_z_t0"] = vae_params["rnn_params"].pop(
                "train_noise_prior_t0"
            )

        if "scalar_noise_x" in vae_params["rnn_params"]:
            if vae_params["rnn_params"]["scalar_noise_x"] == "Cov":
                vae_params["rnn_params"]["noise_x"] = "full"
            elif vae_params["rnn_params"]["scalar_noise_x"] == False:
                vae_params["rnn_params"]["noise_x"] = "diag"
            else:
                vae_params["rnn_params"]["noise_x"] = "scalar"

        if "scalar_noise_z" in vae_params["rnn_params"]:
            if vae_params["rnn_params"]["scalar_noise_z"] == "Cov":
                vae_params["rnn_params"]["noise_z"] = "full"
            elif vae_params["rnn_params"]["scalar_noise_z"] == False:
                vae_params["rnn_params"]["noise_z"] = "diag"
            else:
                vae_params["rnn_params"]["noise_z"] = "scalar"

        if "scalar_noise_z_t0" in vae_params["rnn_params"]:
            if vae_params["rnn_params"]["scalar_noise_z_t0"] == "Cov":
                vae_params["rnn_params"]["noise_z_t0"] = "full"
            elif vae_params["rnn_params"]["scalar_noise_z_t0"] == False:
                vae_params["rnn_params"]["noise_z_t0"] = "diag"
            else:
                vae_params["rnn_params"]["noise_z_t0"] = "scalar"

        if "readout_rates" in vae_params["rnn_params"]:
            vae_params["rnn_params"]["readout_from"] = vae_params["rnn_params"].pop(
                "readout_rates"
            )
        if vae_params["rnn_params"]["readout_from"] == "currents":
            pass
        elif vae_params["rnn_params"]["readout_from"] == "rates":
            pass
        elif vae_params["rnn_params"]["readout_from"] is True:
            vae_params["rnn_params"]["readout_from"] = "rates"
        else:
            vae_params["rnn_params"]["readout_from"] = "z"

        if "observation" not in vae_params["rnn_params"]:
            if vae_params["rnn_params"]["identity_readout"]:
                vae_params["rnn_params"]["observation"] = "one_to_one"
            else:
                vae_params["rnn_params"]["observation"] = "affine"

        if (
            vae_params["rnn_params"]["activation"] == "relu"
            and "clipped" in vae_params["rnn_params"]
            and vae_params["rnn_params"]["clipped"]
        ):
            vae_params["rnn_params"]["activation"] = "clipped_relu"

        if "out_nonlinearity" not in vae_params["rnn_params"]:
            if (
                "observation_likelihood" in training_params
                and training_params["observation_likelihood"] == "Gauss"
            ):
                vae_params["rnn_params"]["obs_nonlinearity"] = "identity"
            elif "obs_rectify" in vae_params:
                vae_params["rnn_params"]["obs_nonlinearity"] = vae_params.pop(
                    "obs_rectify"
                )
            else:
                print("no out nonlinearity found, setting to identity")
                vae_params["rnn_params"]["obs_nonlinearity"] = "identity"
        else:
            vae_params["rnn_params"]["obs_nonlinearity"] = vae_params["rnn_params"].pop(
                "out_nonlinearity"
            )
        if "shared_tau" in vae_params["rnn_params"]:
            vae_params["rnn_params"]["decay"] = vae_params["rnn_params"].pop(
                "shared_tau"
            )
        if "transition" not in vae_params["rnn_params"]:
            if (
                "full_rank" in vae_params["rnn_params"]
                and vae_params["rnn_params"]["full_rank"] == True
            ):
                vae_params["rnn_params"]["transition"] = "full_rank"
            else:
                vae_params["rnn_params"]["transition"] = "low_rank"

        if training_params["loss_f"] == "VGTF":
            training_params["loss_f"] = "smc"
        elif training_params["loss_f"] == "bs_VGTF":
            training_params["loss_f"] = "bs_smc"
        elif training_params["loss_f"] == "opt_VGTF":
            training_params["loss_f"] = "opt_smc"

        if "observation_likelihood" in training_params:
            vae_params["rnn_params"]["obs_likelihood"] = training_params.pop(
                "observation_likelihood"
            )
        if "obs_likelihood" not in vae_params["rnn_params"]:
            vae_params["rnn_params"]["obs_likelihood"] = "Gauss"

        if "sim_v" in training_params:
            vae_params["rnn_params"]["simulate_input"] = training_params.pop("sim_v")
        elif "simulate_input" not in vae_params["rnn_params"]:
            vae_params["rnn_params"]["simulate_input"] = False

    model = VAE(vae_params)

    d = torch.load(state_dict_file_rnn, map_location=torch.device("cpu"))
    if backward_compat:
        # More backwards compatibility
        for key in list(d.keys()):
            d[key.replace("latent_step", "transition")] = d.pop(key)
        if "transition.AW" in list(d.keys()):
            d["transition.decay_param"] = d.pop("transition.AW")
        if "transition.decay" in list(d.keys()):
            d["transition.decay_param"] = d.pop("transition.decay")
        d["transition.decay_param"] = d["transition.decay_param"].view(1)
        if len(d["observation.Bias"].shape) > 1:
            d["observation.Bias"] = d["observation.Bias"].view(
                d["observation.Bias"].shape[1]
            )
        if (
            vae_params["rnn_params"]["observation"] == "one_to_one"
            and len(d["observation.B"].shape) > 1
        ):
            d["observation.B"] = torch.diagonal(
                d["observation.B"] ** 2
            )  # square because prev bug...

        for key in list(d.keys()):
            if key not in model.rnn.state_dict().keys():
                del d[key]
                print("key " + key + " not found in rnn, deleted")

    model.rnn.load_state_dict(d)

    if model.has_encoder and load_encoder:
        d = torch.load(state_dict_file_encoder, map_location=torch.device("cpu"))
        for key in list(d.keys()):
            if key not in model.encoder.state_dict().keys():
                del d[key]
                print("key " + key + " not found in encoder, deleted")
        model.encoder.load_state_dict(d)

    return model, training_params, task_params


class CPU_Unpickler(pickle.Unpickler):
    """
    from https://github.com/pytorch/pytorch/issues/16797#issuecomment-633423219
    """

    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)
