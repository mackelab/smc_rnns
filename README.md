# Inferring stochastic low-rank RNNs with variational sequential monte-carlo / particle filtering
Code accompanying [Inferring stochastic low-rank recurrent neural networks from neural data](https://arxiv.org/abs/2406.16749)

Matthijs Pals, A Erdem Sağtekin, Felix Pei, Manuel Gloeckler and Jakob H Macke, 2024

### Check out the tutorial! 
Create a conda environment:

`cd smc_rnns`

`conda env create -f smc_rnn_env.yml` (on Linux)

`conda enc create -f smc_rnn_env_mac.yml` (on Mac OS)


and open:

 `tutorial/tutorial_continuous.ipynb`


### Finding fixed points
We included code for finding fixed points (and cycles) in piecewise linear low-rank RNNs. The `fixed_points` folder contains scripts for both our 'semi'-analytic method, as well as a modified version of [SCYFI](https://github.com/DurstewitzLab/SCYFI) (Eisenmann et al. 2023, GNU General Public Licence), where the search-space can additionally be constrained to the sub-regions that can contain fixed points. 

Example usage in:
```
generate_figures/Fig_8_find_fixed_points.ipynb
```

### Student-teacher setups
Train the teacher and student networks using:
```
train_scripts/student_teacher/train_student_teacher_continuous.ipynb
train_scripts/student_teacher/train_student_teacher_poisson.ipynb
train_scripts/student_teacher/train_student_teacher_conditioning.ipynb
```
Generate figure:
```
generate_figures/Fig_3_create_figure.ipynb
```


### EEG
We used openly accessible electroencephalogram (EEG) data from Schalk et al. 2004, available from [Physionet](https://www.physionet.org/content/eegmmidb/1.0.0/) (Moody et al. 2000; ODC-BY licence). This repo includes preprocessed data from session S001R01.

Train models using: 
```
python run train_scripts/eeg/run_eeg.py
```
Generate figure and table:
```
generate_figures/Fig_4_plot_EEG.ipynb
generate_figures/Table_1_calc_stats.ipynb
```


### Rodent hippocampal datasets
To run these experiments you first need to obtain the data from CRCNS: [hc2](https://crcns.org/data-sets/hc/hc-2/about-hc-2) (Mizuseki et al. 2009) and [hc11](https://crcns.org/data-sets/hc/hc-11/about-hc-11) (Grosmark et al. 2016)

Data can be preprocessed using the notebooks:
```
train_scripts/hpc/hpc2_spike_preprocessing.ipynb
train_scripts/hpc/hpc1_spike_preprocessing.ipynb
train_scripts/hpc/hpc11_lfp_preprocessing.ipynb
```


Train models using:
```
wandb sweep train_scripts/hpc/sweep.yaml
wandb agent [name]/smc_rnns-train_scripts_hpc/[sweep-id]
```
Generate figures:
```
generate_figures/Fig_5_hpc2.ipynb
generate_figures/Fig_6_S5_hpc11.ipynb
```


### Macaque Reach
To run these experiments you first need to obtain the MC_Maze dataset from the [Neural Latents Benchmark](https://dandiarchive.org/dandiset/000128) (Pei et al. 2021 CC-BY-4.0 licence)

We need to process the dataset and either extract spikes + context input (first line) or just the spikes (second line, for the NLB evaluation) using:
```
python train_scripts/macaque_reach/make_tensors_conditioning.py --binsize 20
python train_scripts/macaque_reach/make_tensors_nlb.py --binsize 20
```

Train models using:
```
python train_scripts/macaque_reach/train_single_conditioning.py --run_name reach_conditioning
python train_scripts/macaque_reach/train_single_nlb.py --run_name reach_nlb 'dataset=mc_maze_20ms_val_nlb'
```

Generate figures:
```
generate_figures/Fig_7_Maze.ipynb
```
-------------------

Feel free to reach out with any questions!
