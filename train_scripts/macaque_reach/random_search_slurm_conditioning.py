import subprocess
import numpy as np
import sys
import time
from pathlib import Path

WRAPPER_SCRIPT = str(Path(__file__).absolute().parent / "run_bash_command.sh")
RUN_SCRIPT = str(Path(__file__).absolute().parent / "train_single_conditioning.py")
RUN_ROOT = str(Path(__file__).absolute().parent.parent.parent.parent / "runs")

# parse search args
parser = argparse.ArgumentParser(description="Train random search on NLB dataset")
parser.add_argument("--run_name", "-r", type=str)
parser.add_argument("--num_models", "-n", type=int, default=10)
args, extras = parser.parse_known_intermixed_args()
run_name = args.run_name
n_runs = args.num_models

# parse default overrides
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

# configure search space
rng = np.random.default_rng(np.random.randint(100))

SEARCH_SPACE = {
    "training_params.lr": lambda: np.power(
        10, rng.uniform(-3.1, -2.75)
    ),  # learning rate
    "training_params.k": lambda: rng.choice([128, 192]),  # number of particles
    "vae_params.dim_z": lambda: rng.choice([5, 6, 8, 16]),  # latent dimensionality
}

# submit jobs
for i in range(n_runs):
    run_overrides = overrides.copy()
    run_overrides.update({k: v() for k, v in SEARCH_SPACE.items()})
    run_overrides["training_params.lr_end"] = run_overrides["training_params.lr"] * 1e-1
    run_override_str = " ".join([f"'--{k}={v}'" for k, v in run_overrides.items()])

    command = (
        f"sbatch -J rs{i:03d} -o {RUN_ROOT}/{run_name}/{i:03d}.log --gres=gpu:1 --ntasks=4 --partition=2080-galvani "
        f"{WRAPPER_SCRIPT} "
        f"{sys.executable} {RUN_SCRIPT} --run_name='{run_name}/{i:03d}' {run_override_str}"
    )

    print(command)
    print(subprocess.getoutput(command))
    # subprocess.getoutput('sleep 600 &')
    # print(i)
    time.sleep(5)
