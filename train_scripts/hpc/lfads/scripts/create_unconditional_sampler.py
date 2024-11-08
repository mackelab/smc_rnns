import hydra
import pickle
import torch
from pathlib import Path

# model paths
config_path = Path("../configs/pbt.yaml")
run_tag = "240802_pbt_con"
# checkpoint_path = sorted(Path(f"~/runs/vi_rnns_hpc2/{run_tag}/best_model/").expanduser().glob("*/tune.ckpt"))[-1]
checkpoint_path = sorted(
    Path(f"~/project/vi_rnns/lfads/vi_rnns_hpc2/{run_tag}/best_model/")
    .expanduser()
    .glob("*/tune.ckpt")
)[-1]

# Compose the train config with properly formatted overrides
DATASET_STR = "vi_rnns_hpc2"
MODEL_STR = "con"
overrides = {
    "datamodule": DATASET_STR,
    "model": MODEL_STR,
}
overrides = [f"{k}={v}" for k, v in (overrides).items()]
config_path = Path(config_path)
with hydra.initialize(
    config_path=config_path.parent,
    job_name="run_model",
    version_base="1.1",
):
    config = hydra.compose(config_name=config_path.name, overrides=overrides)

# load checkpoint
model = hydra.utils.instantiate(config.model)
model.load_state_dict(torch.load(checkpoint_path)["state_dict"])
sampler_state_dict = dict(
    ic_prior=model.ic_prior.state_dict(),
    decoder=model.decoder.state_dict(),
    readout=model.readout[0].state_dict(),
)

if model.co_prior.lognvars.shape[0] > 0:
    sampler_state_dict["co_prior"] = model.co_prior.state_dict()

with open(f"../models/{run_tag}_sampler.pkl", "wb") as f:
    pickle.dump(sampler_state_dict, f)
