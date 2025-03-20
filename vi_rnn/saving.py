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

        # Generate a name
        name = (
            task_params["name"]
            + "_"
            + model.vae_params["enc_architecture"]
            + "_"
            + model.vae_params["rnn_architecture"]
            + "_Z_Date_"
            + str(model.dim_z)
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
    torch.save(model.encoder.state_dict(), state_dict_file_encoder)

    return directory + name


def load_model(name, load_encoder=True):
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

    # Backwards compatibility
    if "prior_params" in vae_params:
        vae_params["rnn_params"] = vae_params.pop("prior_params")
    if "readout_rates" in vae_params["rnn_params"]:
        vae_params["rnn_params"]["readout_from"] = vae_params["rnn_params"].pop(
            "readout_rates"
        )
    if vae_params["rnn_params"]["readout_from"] == True:
        vae_params["rnn_params"]["readout_from"] = "rates"

    if (
        vae_params["rnn_params"]["activation"] == "relu"
        and "clipped" in vae_params
        and vae_params["prior_params"]["clipped"]
    ):
        vae_params["rnn_params"]["activation"] = "clipped_relu"

    if "out_nonlinearity" not in vae_params["rnn_params"]:
        if training_params["observation_likelihood"] == "Gauss":
            vae_params["rnn_params"]["out_nonlinearity"] = "identity"
        elif "obs_rectify" in vae_params:
            vae_params["rnn_params"]["out_nonlinearity"] = vae_params.pop("obs_rectify")
        else:
            print("no out nonlinearity found, setting to identity")
            vae_params["rnn_params"]["out_nonlinearity"] = "identity"

    model = VAE(vae_params)

    # More backwards compatibility
    d = torch.load(state_dict_file_rnn, map_location=torch.device("cpu"))
    for key in list(d.keys()):
        d[key.replace("latent_step", "transition")] = d.pop(key)
    for key in list(d.keys()):
        if key not in model.rnn.state_dict().keys():
            del d[key]
            print("key " + key + " not found in rnn, deleted")
    model.rnn.load_state_dict(d)
    if load_encoder:
        d = torch.load(state_dict_file_encoder, map_location=torch.device("cpu"))
        for key in list(d.keys()):
            if key not in model.encoder.state_dict().keys():
                del d[key]
                print("key " + key + " not found in encoder, deleted")
        model.encoder.load_state_dict(d)

    return model, vae_params, task_params, training_params


class CPU_Unpickler(pickle.Unpickler):
    """
    from https://github.com/pytorch/pytorch/issues/16797#issuecomment-633423219
    """

    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)
