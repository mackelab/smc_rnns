import sys
import numpy as np
import argparse
from pathlib import Path
from nlb_tools.nwb_interface import NWBDataset
from nlb_tools.make_tensors import (
    make_train_input_tensors,
    make_eval_input_tensors,
    make_eval_target_tensors,
)

# -- prepare loading NWB dataset ------------------
DANDI_ROOT = Path(__file__).absolute().parent.parent.parent / "data_untracked" / "dandi"
OUTPUT_ROOT = (
    Path(__file__).absolute().parent.parent.parent / "data_untracked" / "processed"
)
print(f"Looking for files in directory {DANDI_ROOT}")
print(f"Saving files in directory {OUTPUT_ROOT}")
data_map = {
    "mc_maze": DANDI_ROOT / "000128" / "sub-Jenkins",
    "mc_rtt": DANDI_ROOT / "000129" / "sub-Indy",
    "area2_bump": DANDI_ROOT / "000127" / "sub-Han",
    "dmfc_rsg": DANDI_ROOT / "000130" / "sub-Haydn",
    "mc_maze_large": DANDI_ROOT / "000138" / "sub-Jenkins",
    "mc_maze_medium": DANDI_ROOT / "000139" / "sub-Jenkins",
    "mc_maze_small": DANDI_ROOT / "000140" / "sub-Jenkins",
}

# -- args --------------
parser = argparse.ArgumentParser(
    description="Pre-process NWB dataset for model training"
)
parser.add_argument("-d", "--dataset", default="mc_maze", choices=list(data_map.keys()))
parser.add_argument("-p", "--phase", default="val", choices=["val", "test"])
parser.add_argument("-b", "--binsize", type=int, default=5)
args = parser.parse_args()
dataset_name = args.dataset
phase = args.phase
bin_size = args.binsize
print(
    f"Preparing data from {dataset_name} dataset for phase {phase} at bin size {bin_size}"
)

# -- load dataset ---------------
data_dir = data_map[dataset_name]
output_dir = OUTPUT_ROOT / dataset_name / phase
output_dir.mkdir(exist_ok=True, parents=True)
print(f"Reading data from {data_dir}")
dataset = NWBDataset(fpath=str(data_dir))
training_split = "train" if phase == "val" else ["train", "val"]
eval_split = phase

dataset.resample(bin_size)

# -- make and save tensors ---------------------
make_train_input_tensors(
    dataset=dataset,
    dataset_name=dataset_name,
    trial_split=training_split,
    save_file=True,
    save_path=str(output_dir / f"train_input_{bin_size}ms.h5"),
    include_forward_pred=True,
    include_behavior=True,
)

make_eval_input_tensors(
    dataset=dataset,
    dataset_name=dataset_name,
    trial_split=eval_split,
    save_file=True,
    save_path=str(output_dir / f"eval_input_{bin_size}ms.h5"),
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
