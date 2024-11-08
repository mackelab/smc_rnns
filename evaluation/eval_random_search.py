import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import h5py
from sklearn.preprocessing import LabelEncoder
from omegaconf import OmegaConf

from pathlib import Path

# set search dirs
SAVE_NAME = "maze_co_nc"

RUN_DIRS = [
    # Path("/mnt/qb/work/macke/mwe521/vi_rnns/runs/maze_co_rs0_nc"),
    Path("/home/matthijs/runs/reach_nlb/"),  # checkpoints/last/")
]
Path("./plots/").mkdir(exist_ok=True)
Path("./results/").mkdir(exist_ok=True)


# define helper funcs
def h5_to_dict(h5obj):
    data = {}
    for key in h5obj.keys():
        if isinstance(h5obj[key], h5py.Group):
            data[key] = h5_to_dict(h5obj[key])
        else:
            data[key] = h5obj[key][()]
    return data


def compile_configs(configs):
    compiled_raw = {}

    def rget(d: dict, keys: list[str] = []):
        obj = d
        for k in keys:
            obj = obj[k]
        return obj

    def _compile_configs(configs, keys, compiled):
        base = rget(configs[0], keys)
        for k in base.keys():
            if isinstance(base[k], dict):
                _compile_configs(configs, [*keys, k], compiled)
            else:
                values = [rget(cfg, [*keys, k]) for cfg in configs]
                values = (lambda x: [np.nan if a is None else a for a in x])(values)
                if len(np.unique(values, axis=0)) > 1:
                    compiled[".".join([*keys, k])] = values

    _compile_configs(configs, [], compiled_raw)
    labels = dict()
    for key, val in compiled_raw.items():
        key_labels = tuple(np.unique(val, axis=0, return_inverse=True)[1])
        if key_labels in labels:
            labels[key_labels].append(key)
        else:
            labels[key_labels] = [key]
    compiled = compiled_raw.copy()  # TODO implement
    return pd.DataFrame(compiled)


# fetch results for every model
configs = []
results = []
for model_dir in itertools.chain(*[rd.iterdir() for rd in RUN_DIRS]):
    if not model_dir.is_dir():
        continue
    print(model_dir)
    configs.append(OmegaConf.to_object(OmegaConf.load(model_dir / "config.yaml")))
    if (model_dir / "checkpoints" / "best" / "metrics.h5").exists():
        with h5py.File(model_dir / "checkpoints" / "best" / "metrics.h5", "r") as h5f:
            results.append(h5_to_dict(h5f))
    else:
        results.append(dict())
results = pd.DataFrame(results)
configs = compile_configs(configs)
results = pd.concat([configs, results], axis=1)
results.to_csv(f"./results/{SAVE_NAME}.csv")

# plot scatters
# METRIC = "co-bps"
# METRIC_NAME = "co-bps"  # just for plotting, since scaling datasets have extra [500] co-bps etc.
METRIC = "vel R2"
METRIC_NAME = METRIC
fig, axs = plt.subplots(
    1, len(configs.columns), figsize=(len(configs.columns) * 2.4, 2.0), sharey=True
)
for i, config_key in enumerate(configs.columns):
    if isinstance(results[config_key][0], (int, float, np.integer, np.floating)):
        axs[i].scatter(results[config_key], results[METRIC])
    else:
        try:
            encoder = LabelEncoder()
            labels = encoder.fit_transform(results[config_key])
            ticklabels = encoder.inverse_transform(np.arange(np.max(labels) + 1))
        except:
            ticklabels, labels = np.unique(
                list(results[config_key]), axis=0, return_inverse=True
            )
        axs[i].scatter(labels + np.random.randn(*labels.shape) * 0.1, results[METRIC])
        axs[i].set_xticks(np.arange(np.max(labels) + 1), ticklabels)
        axs[i].set_ylim(0, np.max(results[METRIC]) * 1.1)
    axs[i].set_xlabel(config_key)
axs[0].set_ylabel(METRIC_NAME)
plt.tight_layout()
plt.savefig(f"./plots/{SAVE_NAME}.png", dpi=300)
