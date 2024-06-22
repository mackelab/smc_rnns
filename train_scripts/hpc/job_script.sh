#!/bin/bash
#SBATCH --partition=2080-galvani                 # partition
#SBATCH --gres=gpu:1       # type and number of gpus
#SBATCH --time=72:00:00              # job will be cancelled after 6h 30min, ma$
#SBATCH --ntasks=1                # Number of tasks (see below)
#SBATCH --cpus-per-task=10         # Number of CPU cores per task
#SBATCH --mem=12000
#SBATCH --nodes=10                 # Ensure that all cores are on one machine
#SBATCH --output=[LOCATION] vi_%j.out  # File to which STDOUT w$
#SBATCH --error=[LOCATION] vi_%j.err   # File to which STDERR w$
#SBATCH --mail-type=END           # Type of email notification- BEGIN,END,FAIL,$
#SBATCH --mail-user=[MAIL]    # Email to which notifications $

wandb agent vgtfi-rnn/uncategorized/SWEEPID