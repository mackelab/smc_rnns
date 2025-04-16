from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import h5py
from pathlib import Path


class Basic_dataset(Dataset):
    def __init__(self, task_params, data, data_eval=None, stim=None, stim_eval=None):
        """
        Basic dataset class for time series data that returns a random trial of length self.dur
        Args:
            task_params (dict): dictionary of task parameters
            data (np.ndarray; dim_x x T): time series data
            data_eval (np.ndarray; dim_x x T): optional evaluation data
        """
        self.task_params = task_params
        self.data = torch.from_numpy(data)
        if data_eval is not None:
            self.data_eval = torch.from_numpy(data_eval)
        else:
            self.data_eval = self.data
        self.dur = task_params["dur"]
        self.n_trials = task_params["n_trials"]

        if stim is not None:
            self.stim = torch.from_numpy(stim)
        else:
            self.stim = torch.zeros(0, self.data.shape[1])
        if stim_eval is not None:
            self.stim_eval = torch.from_numpy(stim_eval)
        else:
            self.stim_eval = torch.zeros(0, self.data_eval.shape[1])

    def __len__(self):
        """Return number of trials in an epoch"""
        return self.n_trials

    def __getitem__(self, idx):
        """
        Return a trial of length self.dur
        Args:
            idx (int): trial index, arbitrary as trials are sampled randomly
        Returns:
            trial (torch.tensor; dim_x x self.dur): trial of length self.dur
            input (torch.tensor; n_inp x self.dur): optional input on which the model is conditioned
        """
        t_start = torch.randint(low=0, high=self.data.shape[1] - self.dur, size=(1,))[0]
        t_end = t_start + self.dur
        return self.data[:, t_start:t_end], self.stim[:, t_start:t_end]


class Basic_dataset_with_trials(Dataset):
    def __init__(self, task_params, data, data_eval=None, stim=None, stim_eval=None):
        """
        Basic dataset class for time series data split into trials
        Args:
            task_params (dict): dictionary of task parameters
            data (np.ndarray; n_trials x dim_x x T): time series data
            data_eval (np.ndarray; n_trials x dim_x x T): optional evaluation data
            inputs (np.ndarray; n_trials x dim_u x T): optional input
        """
        self.task_params = task_params
        self.data = torch.from_numpy(data)
        if data_eval is not None:
            self.data_eval = torch.from_numpy(data_eval)
        else:
            self.data_eval = self.data

        self.n_trials = self.data.shape[0]

        if stim is not None:
            self.stim = torch.from_numpy(stim)
        else:
            self.stim = torch.zeros(self.n_trials, 0, self.data.shape[2])
        if stim_eval is not None:
            self.stim_eval = torch.from_numpy(stim_eval)
        else:
            self.stim_eval = torch.zeros(
                self.data_eval.shape[0], 0, self.data_eval.shape[2]
            )

    def __len__(self):
        """Return number of trials in an epoch"""
        return self.n_trials

    def __getitem__(self, idx):
        """
        Return a trial of length self.dur
        Args:
            idx (int): trial index, arbitrary as trials are sampled randomly
        Returns:
            trial (torch.tensor; dim_x x self.dur): trial of length self.dur
            input (torch.tensor; n_inp x self.dur): optional input on which the model is conditioned
        """
        return self.data[idx], self.stim[idx]


class Oscillations_Cont(Dataset):
    def __init__(self, task_params, U, V, B, decay=0.9):
        """
        Teacher student dataset class for continuous oscillations
        Args:
            task_params (dict): dictionary of task parameters
            U (torch.tensor; N x R): Left singular vectors
            V (torch.tensor; R x N): (Scaled) Right singular vectors
            B (torch.tensor; N): biases
            decay (float): decay rate
        Initialises:
            self.data (torch.tensor; N_trials x self.dim_x x T): data
            self.latents (torch.tensor; N_trials x self.dim_z x T): latents
        """

        self.dur = task_params["dur"]
        self.n_trials = task_params["n_trials"]
        self.N = task_params["n_neurons"]
        self.non_lin = task_params["non_lin"]
        self.R_z = task_params["R_z"]
        self.R_x = task_params["R_x"]
        self.task_params = task_params
        if "ph0" in task_params and task_params["ph0"] is not None:
            ph0 = task_params["ph0"]
        else:
            ph0 = torch.randn(self.n_trials) * np.pi * 2
        if "r0" in task_params:
            r0 = task_params["r0"]
        else:
            r0 = torch.randn(self.n_trials) * 2
        self.latents = torch.zeros(self.n_trials, 2, self.dur, dtype=torch.float32)
        self.data = torch.zeros(self.n_trials, self.N, self.dur, dtype=torch.float32)
        self.latents[:, 0, 0] = r0 * np.cos(ph0)
        self.latents[:, 1, 0] = r0 * np.sin(ph0)
        self.latents[:, :, 0] += torch.randn(self.n_trials, 2) * self.R_z

        for t in range(1, self.dur):
            self.latents[:, :, t] += decay * self.latents[:, :, t - 1]
            self.latents[:, :, t] += (
                self.non_lin(self.latents[:, :, t - 1] @ U.T + B.unsqueeze(0)) @ V.T
                + torch.randn(self.n_trials, 2) * self.R_z
            )

        if task_params["out"] == "rates":
            for t in range(self.dur):
                self.data[:, :, t] = self.non_lin(
                    self.latents[:, :, t] @ U.T + B.unsqueeze(0)
                )[:, : self.N]
        elif task_params["out"] == "currents":
            for t in range(self.dur):
                self.data[:, :, t] = (self.latents[:, :, t] @ U.T)[:, : self.N]
        self.data += torch.randn(self.n_trials, self.N, self.dur) * self.R_x
        self.data_eval = self.data
        self.stim = torch.zeros(self.data.shape[0], 0, self.data.shape[2])
        self.stim_eval = self.stim

    def __len__(self):
        """Return number of trials in an epoch"""
        return self.n_trials

    def __getitem__(self, idx):
        """
        Return a trial of length self.dur
        Args:
            idx (int): trial index
        Returns:
            trial (torch.tensor; self.dur x dim_x): trial of length self.dur
            input (torch.tensor; self.dur x dim_u): optional input (here 0)
        """

        return self.data[idx], self.stim[idx]


class Oscillations_Poisson(Dataset):
    def __init__(self, task_params, U, V, B, decay=0.9):
        """
        Teacher student dataset class for oscillations with Poisson observations
        Args:
            task_params (dict): dictionary of task parameters
            U (torch.tensor; N x R): Left singular vectors
            V (torch.tensor; R x N): (Scaled) Right singular vectors
            B (torch.tensor; N): biases
            decay (float): decay rate
        Initialises:
            self.data (torch.tensor N_trials x self.dim_x x T): Poisson spikes
            self.rates (torch.tensor N_trials x self.dim_x x T): Poisson rates
            self.latents (torch.tensor N_trials x self.dim_z x T): latents
        """

        self.dur = task_params["dur"]
        self.n_trials = task_params["n_trials"]
        self.N = task_params["n_neurons"]
        self.non_lin = task_params["non_lin"]
        self.R_z = task_params["R_z"]
        self.task_params = task_params

        if "ph0" in task_params and task_params["ph0"] is not None:
            ph0 = task_params["ph0"]
        else:
            ph0 = torch.randn(self.n_trials) * np.pi * 2
        if "r0" in task_params:
            r0 = task_params["r0"]
        else:
            r0 = torch.randn(self.n_trials) * 2
        self.latents = torch.zeros(self.n_trials, 2, self.dur, dtype=torch.float32)
        self.rates = torch.zeros(self.n_trials, self.N, self.dur, dtype=torch.float32)
        self.latents[:, 0, 0] = r0 * np.cos(ph0)
        self.latents[:, 1, 0] = r0 * np.sin(ph0)
        self.latents[:, :, 0] += torch.randn(self.n_trials, 2) * self.R_z

        for t in range(1, self.dur):
            self.latents[:, :, t] += decay * self.latents[:, :, t - 1]
            self.latents[:, :, t] += (
                self.non_lin(self.latents[:, :, t - 1] @ U.T + B.unsqueeze(0)) @ V.T
                + torch.randn(self.n_trials, 2) * self.R_z
            )

        if task_params["out"] == "rates":
            for t in range(self.dur):
                self.rates[:, :, t] = self.non_lin(
                    self.latents[:, :, t] @ U.T + B.unsqueeze(0)
                )[:, : self.N]
        elif task_params["out"] == "currents":
            for t in range(self.dur):
                self.rates[:, :, t] = (self.latents[:, :, t] @ U.T)[:, : self.N]

        self.rates *= task_params["B"]
        self.rates += task_params["Bias"]

        if task_params["obs_rectify"] == "exp":
            self.rates = torch.exp(self.rates)
        elif task_params["obs_rectify"] == "relu":
            self.rates = torch.relu(self.rates) + 1e-10
        elif task_params["obs_rectify"] == "softplus":
            self.rates = torch.nn.functional.softplus(self.rates)
        self.data = torch.poisson(self.rates)
        self.data_eval = self.data
        self.stim = torch.zeros(self.data.shape[0], 0, self.data.shape[2])
        self.stim_eval = self.stim

    def __len__(self):
        """Return number of trials in an epoch"""
        return self.n_trials

    def __getitem__(self, idx):
        """
        Return a trial of length self.dur
        Args:
            idx (int): trial index
        Returns:
            trial (torch.tensor; self.dur x dim_x): trial of length self.dur
            input (torch.tensor; self.dur x dim_u): optional input (here 0)
        """

        return self.data[idx], self.stim[idx]


class Reaching_Teacher(Dataset):
    def __init__(self, task_params, task_params_teacher, U, V, B, I, decay=0.9):
        """
        Initialize Teacher data for Reaching task
        Args:
            task_params: dictionary containing task parameters
            task_params_teacher: dictionary containing teacher task parameters
            U: torch.tensor; N x R, input to hidden weights
            V: torch.tensor; R x N, hidden to hidden weights
            B: torch.tensor; N, biases
            I: torch.tensor; n_inp x N, input to hidden weights
            decay: float, decay rate
        Initialises:
            self.data (torch.tensor N_trials x self.dim_x x T): data
            self.v (torch.tensor N_trials x self.dim_u x T): input filtered by timeconstant
            self.latents (torch.tensor N_trials x self.dim_z x T ):  latents
        """

        self.dur = task_params["dur"]
        self.n_trials = task_params["n_trials"]
        self.N = task_params["n_neurons"]
        self.w = task_params["w"]
        self.non_lin = task_params["non_lin"]
        self.R_z = task_params["R_z"]
        self.R_x = task_params["R_x"]
        self.task_params = task_params

        n_repeats = task_params["n_trials"] // task_params_teacher["n_stim"]

        if task_params["n_trials"] % task_params_teacher["n_stim"] != 0:
            n_repeats += 1

        # obtain teacher RNNs stimuli
        reaching = Reaching(task_params_teacher)
        Reaching_loader = DataLoader(
            reaching, batch_size=task_params_teacher["n_stim"], shuffle=False
        )
        s, _, _ = next(iter(Reaching_loader))  # x = trial,time,stims
        s = s.repeat(n_repeats, 1, 1).permute(0, 2, 1)[: task_params["n_trials"]]
        print(s.shape)
        self.stim = s

        # generate teacher data
        if "ph0" in task_params and task_params["ph0"] is not None:
            ph0 = task_params["ph0"]
        else:
            ph0 = torch.randn(self.n_trials) * np.pi * 2
        if "r0" in task_params:
            r0 = task_params["r0"]
        else:
            r0 = torch.randn(self.n_trials) * 0.1

        self.latents = torch.zeros(self.n_trials, 2, self.dur, dtype=torch.float32)
        self.data = torch.zeros(self.n_trials, self.N, self.dur, dtype=torch.float32)
        self.latents[:, 0, 0] = r0 * np.cos(ph0)
        self.latents[:, 1, 0] = r0 * np.sin(ph0)
        self.latents[:, :, 0] += torch.randn(self.n_trials, 2) * self.R_z

        # input filtered by timeconstant
        self.v = torch.zeros(self.n_trials, 2, self.dur, dtype=torch.float32)

        for t in range(1, self.dur):
            if task_params["sim_v"] == True:
                self.v[:, :, t] = decay * self.v[:, :, t - 1] + (1 - decay) * (
                    s[:, :, t - 1]
                )
            else:
                self.v[:, :, t] = s[:, :, t]
            self.latents[:, :, t] = decay * self.latents[:, :, t - 1]
            X = self.latents[:, :, t - 1] @ U.T + self.v[:, :, t - 1] @ I
            self.latents[:, :, t] += (
                self.non_lin(X + B.unsqueeze(0)) @ V.T
                + torch.randn(self.n_trials, 2) * self.R_z
            )

        if task_params["out"] == "rates":
            for t in range(self.dur):
                self.data[:, :, t] = self.non_lin(
                    self.latents[:, :, t] @ U.T
                    + self.v[
                        :,
                        t:,
                    ]
                    @ I
                    + B.unsqueeze(0)
                )
        elif task_params["out"] == "currents":
            for t in range(self.dur):
                self.data[:, :, t] = self.latents[:, :, t] @ U.T + self.v[:, :, t] @ I
        self.data += torch.randn(self.n_trials, self.N, self.dur) * self.R_x
        self.data_eval = self.data
        self.stim_eval = self.stim

    def __len__(self):
        """Return number of trials in an epoch"""
        return self.n_trials

    def __getitem__(self, idx):
        """
        Return a trial of length self.dur
        Args:
            idx (int): trial index
        Returns:
            trial (torch.tensor; dim_x x self.dur): trial of length self.dur
            stim (torch.tensor; n_inp x self.dur): stimulus
        """
        return self.data[idx], self.stim[idx]


class Reaching(Dataset):
    def __init__(self, task_params):
        """
        Initialize a Reaching task (for teacher network)
        Args:
            task_params: dictionary containing task parameters
        """

        self.task_params = task_params

    def __len__(self):
        """Amount of reach directions"""
        return self.task_params["n_stim"]

    def __getitem__(self, idx):
        """
        Returns a trial

        Args:
            idx, trial index

        Returns:
            input, Tensor of size [seq_len, n_inp]
            target, Tensor of size [seq_len, n_inp]
            mask, Tensor of size [seq_len, n_inp]
        """
        phase = np.pi * 2 * idx / self.task_params["n_stim"]
        inputs = torch.zeros(self.task_params["trial_len"], 2)

        # give input as sine and cosine of angle
        cp = np.cos(phase)
        sp = np.sin(phase)
        onset = torch.randint(
            low=self.task_params["onset"][0],
            high=self.task_params["onset"][1],
            size=(1,),
        )[0]
        stim_dur = torch.randint(
            low=self.task_params["stim_dur"][0],
            high=self.task_params["stim_dur"][1],
            size=(1,),
        )[0]

        target_start = onset + stim_dur
        inputs[onset:target_start, 0] = cp
        inputs[onset:target_start, 1] = sp

        targets = torch.zeros(self.task_params["trial_len"], 2)
        targets[target_start:, 0] = cp
        targets[target_start:, 1] = sp

        # mask out the stimulus period for loss
        mask = torch.zeros_like(targets)
        mask[target_start:] = 1
        return inputs, targets, mask


class SineWave(Dataset):
    def __init__(self, task_params):
        """
        Initialize a Sinewave task (for teacher network)
        Args:
            task_params: dictionary containing task parameters
        """

        self.task_params = task_params

    def __len__(self):
        """Arbitrary number of trials, as they are randomly generated anyway"""
        return self.task_params["n_trials"]

    def __getitem__(self, idx):
        """
        Returns a trial

        Args:
            idx, trial index

        Returns:
            input, Tensor of size [seq_len, n_inp]
            target, Tensor of size [seq_len, n_inp]
            mask, Tensor of size [seq_len, n_inp]
        """
        inputs = torch.zeros(self.task_params["dur"], 1)
        targets = np.sin(
            torch.linspace(
                0, self.task_params["n_cycles"] * 2 * np.pi, self.task_params["dur"]
            )
        ).unsqueeze(1)
        mask = torch.ones(self.task_params["dur"], 1)
        return inputs, targets, mask


class RDM(Dataset):
    def __init__(self, task_params):
        """
        Initialize a random dot motion / perceptual decision making
         / evidence integration task (for teacher network)
        Args:
            task_params: dictionary containing task parameters

        Based on https://github.com/adrian-valente/populations_paper_code/blob/master/low_rank_rnns/rdm.py
        Dubreuil A., Valente A., Beiran M., Mastrogiuseppe F., Ostojic S.
        The role of population structure in computations through neural dynamics
        Nature Neuroscience volume 25, pages 783–794 (2022)
        """

        self.task_params = task_params

        if "coherences" not in task_params or task_params["coherences"] is None:
            self.coherences = [-4, -2, -1, 1, 2, 4]
        else:
            self.coherences = task_params["coherences"]

        self.fixation_dur = task_params["fixation_dur"]
        self.stimulus_end = task_params["stimulus_dur"] + task_params["fixation_dur"]
        self.response_begin = self.stimulus_end + task_params["delay_dur"]
        self.total_duration = self.response_begin + task_params["response_dur"]

        self.std = task_params["std"]
        self.scale = task_params["scale"]

    def __len__(self):
        return len(self.coherences)

    def __getitem__(self, idx):
        """
        Returns a trial

        Args:
            idx, trial index

        Returns:
            input, Tensor of size [seq_len, n_inp]
            target, Tensor of size [seq_len, n_inp]
            mask, Tensor of size [seq_len, n_inp]
        """

        inputs = self.std * torch.randn((self.total_duration, 1), dtype=torch.float32)
        targets = torch.zeros((self.total_duration, 1), dtype=torch.float32)
        mask = torch.zeros((self.total_duration, 1), dtype=torch.float32)

        coh_current = self.coherences[idx]
        inputs[self.fixation_dur : self.stimulus_end] += coh_current * self.scale
        targets[self.response_begin :] = np.sign(coh_current)
        mask[self.response_begin :] = 1
        return inputs, targets, mask


class RDM_Teacher(Dataset):
    def __init__(self, task_params, task_params_teacher, U, V, B, I, decay=0.9):
        """
        Initialize Teacher data for Reaching task
        Args:
            task_params: dictionary containing task parameters
            task_params_teacher: dictionary containing teacher task parameters
            U: torch.tensor; N x R, input to hidden weights
            V: torch.tensor; R x N, hidden to hidden weights
            B: torch.tensor; N, biases
            I: torch.tensor; 1 x N, input to hidden weights
            decay: float, decay rate
        Initialises:
            self.data (torch.tensor N_trials x self.dim_x x T): data
            self.v (torch.tensor N_trials x self.dim_u x T: input filtered by timeconstant
            self.latents (torch.tensor N_trials x self.dim_z x T):  latents
        """
        self.R_z = task_params["R_z"]
        self.R_x = task_params["R_x"]
        self.task_params = task_params
        # obtain teacher RNNs stimuli
        reaching = RDM(task_params_teacher)
        Reaching_loader = DataLoader(
            reaching, batch_size=reaching.__len__(), shuffle=False
        )
        n_repeats = task_params["n_trials"] // reaching.__len__()
        if task_params["n_trials"] % reaching.__len__() != 0:
            n_repeats += 1
        ss = []
        for _ in range(n_repeats):
            s, _, _ = next(iter(Reaching_loader))  # x = trial,time,stims
            ss.append(s)
        self.stim = torch.concatenate(ss)[: task_params["n_trials"]].permute(0, 2, 1)
        self.dur = self.stim.shape[2]
        self.n_trials = self.stim.shape[0]
        self.N = U.shape[0]
        self.non_lin = torch.nn.ReLU()

        # generate teacher data
        if "r0" in task_params:
            r0 = task_params["r0"]
        else:
            r0 = torch.randn(self.n_trials) * 0.1
        self.latents = torch.zeros(self.n_trials, 1, self.dur, dtype=torch.float32)
        self.data = torch.zeros(self.n_trials, self.N, self.dur, dtype=torch.float32)
        self.latents[:, 0, 0] = r0
        self.latents[:, 0, 0] += torch.randn(self.n_trials) * self.R_z
        self.v = torch.zeros(self.n_trials, 1, self.dur, dtype=torch.float32)
        for t in range(1, self.dur):
            if task_params["sim_v"] == True:
                self.v[:, :, t] = decay * self.v[:, :, t - 1] + (1 - decay) * (
                    self.stim[:, :, t - 1]
                )
            else:
                self.v[:, :, t] = self.stim[:, :, t]
            self.latents[:, :, t] = decay * self.latents[:, :, t - 1]
            X = self.latents[:, :, t - 1] @ U.T + self.v[:, :, t - 1] @ I
            self.latents[:, :, t] += (
                self.non_lin(X + B.unsqueeze(0)) @ V.T
                + torch.randn(self.n_trials, 1) * self.R_z
            )

        if task_params["out"] == "rates":
            for t in range(self.dur):
                self.data[:, :, t] = self.non_lin(
                    self.latents[:, :, t] @ U.T + self.v[:, :, t] @ I + B.unsqueeze(0)
                )
        elif task_params["out"] == "currents":
            for t in range(self.dur):
                self.data[:, :, t] = self.latents[:, :, t] @ U.T + self.v[:, :, t] @ I
        self.data += torch.randn(self.n_trials, self.N, self.dur) * self.R_x
        self.data_eval = self.data
        self.stim_eval = self.stim

    def __len__(self):
        """Return number of trials in an epoch"""
        return self.n_trials

    def __getitem__(self, idx):
        """
        Return a trial of length self.dur
        Args:
            idx (int): trial index
        Returns:
            trial (torch.tensor; dim_x x self.dur): trial of length self.dur
            stim (torch.tensor; n_inp x self.dur): stimulus
        """
        return self.data[idx], self.stim[idx]


class NLBDataset(Dataset):
    def __init__(self, data, params, inputs=None):
        """
        Initialize Teacher data for Reaching task
        Args:
            data (torch.tensor ;N_trials x self.dim_x x T): data
            params (dict): dictionary containing task parameters
            inputs (None or torch.tensor; N_trials x self.dim_u x T: optional stimulus input
        """

        self.data = data
        self.data_eval = data  # Not used in this setup
        self.trial_dur = data.shape[0]
        self.task_params = params
        self.stim = (
            inputs.float().to(self.data.device)
            if inputs is not None
            else torch.zeros(
                data.shape[0],
                0,
                data.shape[2],
                device=self.data.device,
                dtype=torch.float,
            )
        )
        self.stim_eval = self.stim

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.stim[idx]


def load_nlb_dataset(
    data_root: Path,
    name: str,
    phase: str = "val",
    bin_size: int = 5,
    t_forward: int = 0,
    input_field: str = None,
    smooth_input: int = 0,
    u: int = 0,
    cosmooth: bool = True,
    **kwargs,
):
    data_path = data_root / name / phase
    train_inputs = eval_inputs = None
    normalization = None
    with h5py.File(data_path / f"train_input_{bin_size}ms.h5", "r") as h5f:
        if t_forward > 0:
            train_data = np.concatenate(
                [
                    np.concatenate(
                        [
                            h5f["train_spikes_heldin"][()],
                            h5f["train_spikes_heldin_forward"][()],
                        ],
                        axis=1,
                    ),
                    np.concatenate(
                        [
                            h5f["train_spikes_heldout"][()],
                            h5f["train_spikes_heldout_forward"][()],
                        ],
                        axis=1,
                    ),
                ],
                axis=2,
            )
        else:
            train_data = np.concatenate(
                [h5f["train_spikes_heldin"][()], h5f["train_spikes_heldout"][()]],
                axis=2,
            )
        if input_field is not None:
            train_inputs = h5f[f"train_{input_field}"][()]
            if len(train_inputs.shape) == 1:
                train_inputs = train_inputs[:, None]
            if len(train_inputs.shape) == 2:
                train_inputs = np.tile(
                    train_inputs[:, None, :], reps=(1, train_data.shape[1], 1)
                )
            if train_inputs.shape[-1] > u:
                train_inputs = train_inputs[:, :, :u]
            normalization = np.max(np.abs(train_inputs), axis=(0, 1), keepdims=True)
            train_inputs = train_inputs / normalization
            if smooth_input > 1:
                orig_len = train_inputs.shape[1]
                train_inputs = np.pad(
                    train_inputs,
                    ((0, 0), (smooth_input // 2, smooth_input // 2), (0, 0)),
                    mode="edge",
                )
                smoothing_func = lambda x: np.convolve(
                    x, np.ones(smooth_input) / smooth_input, mode="valid"
                )
                train_inputs = np.apply_along_axis(smoothing_func, 1, train_inputs)
                assert train_inputs.shape[1] == orig_len
    with h5py.File(data_path / f"eval_input_{bin_size}ms.h5", "r") as h5f:
        if cosmooth:
            eval_data = h5f["eval_spikes_heldin"][()]
        else:
            assert "eval_spikes_heldout" in h5f.keys()
            eval_data = np.concatenate(
                [h5f["eval_spikes_heldin"][()], h5f["eval_spikes_heldout"][()]], axis=2
            )
        if input_field is not None:
            eval_inputs = h5f[f"eval_{input_field}"][()]
            if len(eval_inputs.shape) == 1:
                eval_inputs = eval_inputs[:, None]
            if len(eval_inputs.shape) == 2:
                eval_inputs = np.tile(
                    eval_inputs[:, None, :], reps=(1, eval_data.shape[1], 1)
                )
            if eval_inputs.shape[-1] > u:
                eval_inputs = eval_inputs[:, :, :u]
            assert normalization is not None
            eval_inputs = eval_inputs / normalization
            if smooth_input > 1:
                orig_len = eval_inputs.shape[1]
                eval_inputs = np.pad(
                    eval_inputs,
                    ((0, 0), (smooth_input // 2, smooth_input // 2), (0, 0)),
                    mode="edge",
                )
                smoothing_func = lambda x: np.convolve(
                    x, np.ones(smooth_input) / smooth_input, mode="valid"
                )
                eval_inputs = np.apply_along_axis(smoothing_func, 1, eval_inputs)
                assert eval_inputs.shape[1] == orig_len
    return (
        torch.tensor(train_data, dtype=torch.float).permute(0, 2, 1),
        torch.tensor(eval_data, dtype=torch.float).permute(0, 2, 1),
        (
            torch.tensor(train_inputs, dtype=torch.float).permute(0, 2, 1)
            if train_inputs is not None
            else train_inputs
        ),
        (
            torch.tensor(eval_inputs, dtype=torch.float).permute(0, 2, 1)
            if eval_inputs is not None
            else eval_inputs
        ),
    )
