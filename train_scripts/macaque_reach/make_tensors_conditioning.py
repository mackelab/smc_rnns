import sys
import numpy as np
import argparse
from pathlib import Path
from nlb_tools.nwb_interface import NWBDataset
from nlb_tools.make_tensors import (
    make_train_input_tensors,
    make_eval_input_tensors,
    make_eval_target_tensors,
    save_to_h5,
)

# -- prepare loading NWB dataset ------------------
DANDI_ROOT = Path(__file__).absolute().parent.parent.parent / "data_untracked" / "dandi"
OUTPUT_ROOT = (
    Path(__file__).absolute().parent.parent.parent / "data_untracked" / "processed"
)
data_map = {
    "mc_maze": DANDI_ROOT / "000128" / "sub-Jenkins",
    "mc_maze_large": DANDI_ROOT / "000138" / "sub-Jenkins",
    "mc_maze_medium": DANDI_ROOT / "000139" / "sub-Jenkins",
    "mc_maze_small": DANDI_ROOT / "000140" / "sub-Jenkins",
}
print(f"Looking for files in directory {DANDI_ROOT}")
print(f"Saving files in directory {OUTPUT_ROOT}")

# -- args ------------
parser = argparse.ArgumentParser(
    description="Pre-process NWB dataset for model training"
)
parser.add_argument("-d", "--dataset", default="mc_maze", choices=list(data_map.keys()))
parser.add_argument("-b", "--binsize", type=int, default=5)
args = parser.parse_args()
dataset_name = args.dataset
phase = "val"  # conditioning not available for test data
bin_size = args.binsize
input_mode = "pos"
print(f"Preparing data from {dataset_name} dataset at bin size {bin_size}")

# -- load data --------------------
data_dir = data_map[dataset_name]
output_dir = OUTPUT_ROOT / f"{dataset_name}_input" / input_mode
output_dir.mkdir(exist_ok=True, parents=True)
print(f"Reading data from {data_dir}")
dataset = NWBDataset(fpath=str(data_dir))
training_split = "train" if phase == "val" else ["train", "val"]
eval_split = phase
dataset.resample(bin_size)

# -- make inputs -------------------
dataset.trial_info = dataset.trial_info[dataset.trial_info.trial_version == 0]
assert np.all(dataset.trial_info.num_barriers == 0)

trial_time = int(round(700 / bin_size))
train_inputs = []
val_inputs = []
for i, trial in dataset.trial_info.iterrows():
    if trial.split == "test":
        continue
    final_target_pos = trial.target_pos[int(round(trial.active_target))]
    final_target_pos = np.array(final_target_pos).squeeze()[None, :]
    trial_input = np.zeros((trial_time, 2))
    trial_input[:, :2] = final_target_pos
    if trial.split == "train":
        train_inputs.append(trial_input)
    elif trial.split == "val":
        val_inputs.append(trial_input)
train_inputs = np.stack(train_inputs, axis=0)
val_inputs = np.stack(val_inputs, axis=0)

# -- make and save tensors
data_dict = make_train_input_tensors(
    dataset=dataset,
    dataset_name=dataset_name,
    trial_split=training_split,
    save_file=False,
    include_forward_pred=True,
    include_behavior=True,
)
data_dict["train_input"] = train_inputs
save_to_h5(
    data_dict,
    str(output_dir / f"train_input_{bin_size}ms.h5"),
    overwrite=True,
)

data_dict = make_eval_input_tensors(
    dataset=dataset,
    dataset_name=dataset_name,
    trial_split=eval_split,
    save_file=False,
)
data_dict["eval_input"] = val_inputs
save_to_h5(
    data_dict,
    str(output_dir / f"eval_input_{bin_size}ms.h5"),
    overwrite=True,
)

if phase == "val":
    make_eval_target_tensors(
        dataset=dataset,
        dataset_name=dataset_name,
        train_trial_split="train",
        eval_trial_split="val",
        include_psth=True,
        save_file=True,
        save_path=str(output_dir / f"eval_target_{bin_size}ms.h5"),
    )
