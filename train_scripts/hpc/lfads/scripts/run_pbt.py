import os
import shutil
import inspect
from datetime import datetime
from pathlib import Path

from ray import tune
from ray.tune import CLIReporter
from ray.tune.search.basic_variant import BasicVariantGenerator

from lfads_torch.extensions.tune import (
    BinaryTournamentPBT,
    HyperParam,
    ImprovementRatioStopper,
)
from lfads_torch.run_model import run_model

# ---------- OPTIONS ----------
DATASET_STR = "vi_rnns_hpc2"
MODEL_STR = "large"
RUN_TAG = datetime.now().strftime("%y%m%d") + f"_pbt_{MODEL_STR}"
RUN_DIR = Path("~/runs").expanduser() / DATASET_STR / RUN_TAG
HYPERPARAM_SPACE = {
    "model.lr_init": HyperParam(
        1e-5, 5e-3, explore_wt=0.3, enforce_limits=True, init=4e-3
    ),
    "model.dropout_rate": HyperParam(
        0.0, 0.6, explore_wt=0.3, enforce_limits=True, sample_fn="uniform"
    ),
    "model.train_aug_stack.transforms.0.cd_rate": HyperParam(
        0.01, 0.7, explore_wt=0.3, enforce_limits=True, init=0.5, sample_fn="uniform"
    ),
    "model.kl_co_scale": HyperParam(1e-6, 1e-4, explore_wt=0.8),
    "model.kl_ic_scale": HyperParam(1e-6, 1e-3, explore_wt=0.8),
    "model.l2_gen_scale": HyperParam(1e-4, 1e-0, explore_wt=0.8),
    "model.l2_con_scale": HyperParam(1e-4, 1e-0, explore_wt=0.8),
}
# ------------------------------

# weird path handling so we can have configs separate from lfads-torch
func_path = Path(inspect.getfile(run_model)).resolve().parent
config_path = Path("../configs/pbt.yaml").resolve()
rel_config_path = os.path.relpath(config_path, func_path)


# Function to keep dropout and CD rates in-bounds
def clip_config_rates(config):
    return {k: min(v, 0.99) if "_rate" in k else v for k, v in config.items()}


init_space = {name: tune.sample_from(hp.init) for name, hp in HYPERPARAM_SPACE.items()}
# Set the mandatory config overrides to select datamodule and model
mandatory_overrides = {
    "datamodule": DATASET_STR,
    "model": MODEL_STR,
}
RUN_DIR.mkdir(parents=True)
# Copy this script into the run directory
shutil.copyfile(__file__, RUN_DIR / Path(__file__).name)
# Run the hyperparameter search
metric = "valid/recon_smth"
num_trials = 20
perturbation_interval = 25
burn_in_period = 80 + 25
analysis = tune.run(
    tune.with_parameters(
        run_model,
        config_path=rel_config_path,
        do_posterior_sample=False,
    ),
    metric=metric,
    mode="min",
    name=RUN_DIR.name,
    stop=ImprovementRatioStopper(
        num_trials=num_trials,
        perturbation_interval=perturbation_interval,
        burn_in_period=burn_in_period,
        metric=metric,
        patience=4,
        min_improvement_ratio=5e-4,
    ),
    config={**mandatory_overrides, **init_space},
    resources_per_trial=dict(cpu=3, gpu=0.5),
    num_samples=num_trials,
    local_dir=RUN_DIR.parent,
    search_alg=BasicVariantGenerator(random_state=0),
    scheduler=BinaryTournamentPBT(
        perturbation_interval=perturbation_interval,
        burn_in_period=burn_in_period,
        hyperparam_mutations=HYPERPARAM_SPACE,
    ),
    keep_checkpoints_num=1,
    verbose=1,
    progress_reporter=CLIReporter(
        metric_columns=[metric, "cur_epoch"],
        sort_by_metric=True,
    ),
    trial_dirname_creator=lambda trial: str(trial),
)
# Copy the best model to a new folder so it is easy to identify
best_model_dir = RUN_DIR / "best_model"
shutil.copytree(analysis.best_logdir, best_model_dir)
# Switch working directory to this folder (usually handled by tune)
os.chdir(best_model_dir)
# Load the best model and run posterior sampling (skip training)
best_ckpt_dir = best_model_dir / Path(analysis.best_checkpoint._local_path).name
run_model(
    overrides=mandatory_overrides,
    checkpoint_dir=best_ckpt_dir,
    config_path=rel_config_path,
    do_train=False,
)

# save everything necessary for sampler
import hydra
import torch
import pickle
from glob import glob

# Compose the train config with properly formatted overrides
overrides = [f"{k}={v}" for k, v in (mandatory_overrides).items()]
config_path = Path(rel_config_path)
with hydra.initialize(
    config_path=config_path.parent,
    job_name="run_model",
    version_base="1.1",
):
    config = hydra.compose(config_name=config_path.name, overrides=overrides)
model = hydra.utils.instantiate(config.model)
ckpt_pattern = os.path.join(best_ckpt_dir, "*.ckpt")
ckpt_path = max(glob(ckpt_pattern), key=os.path.getctime)
model.load_state_dict(torch.load(ckpt_path)["state_dict"])
sampler_state_dict = dict(
    ic_prior=model.ic_prior.state_dict(),
    decoder=model.decoder.state_dict(),
    readout=model.readout[0].state_dict(),
)
with open(best_model_dir / "lfads_sampler.pkl", "wb") as f:
    pickle.dump(sampler_state_dict, f)
