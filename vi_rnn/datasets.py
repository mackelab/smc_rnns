from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import h5py
from pathlib import Path


class Basic_dataset(Dataset):
    def __init__(self, task_params, data, data_eval=None):
        """
        Basic dataset class for time series data that returns a random trial of length self.dur
        Args:
            task_params (dict): dictionary of task parameters
            data (np.ndarray; T x dim_x): time series data
            data_eval (np.ndarray; T x dim_x): optional evaluation data
        """
        self.task_params = task_params
        self.data = torch.from_numpy(data)
        if data_eval is not None:
            self.data_eval = torch.from_numpy(data_eval)
        else:
            self.data_eval = self.data
        self.dur = task_params["dur"]
        self.n_trials = task_params["n_trials"]

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
        t_start = torch.randint(low=0, high=self.data.shape[0] - self.dur, size=(1,))[0]
        t_end = t_start + self.dur
        return self.data[t_start:t_end].T, torch.zeros(
            0, self.dur, device=self.data.device
        )


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

        """

        self.dur = task_params["dur"]
        self.n_trials = task_params["n_trials"]
        self.N = task_params["n_neurons"]
        self.w = task_params["w"]
        self.non_lin = task_params["non_lin"]
        self.R_z = task_params["R_z"]
        self.R_x = task_params["R_x"]

        ph0 = torch.randn(self.n_trials) * np.pi * 2
        if "r0" in task_params:
            r0 = task_params["r0"]
        else:
            r0 = torch.randn(self.n_trials) * 2
        self.latents = torch.zeros(2, self.n_trials, self.dur, dtype=torch.float32)
        self.data = torch.zeros(self.N, self.n_trials, self.dur, dtype=torch.float32)
        self.latents[0, :, 0] = r0 * np.cos(ph0)
        self.latents[1, :, 0] = r0 * np.sin(ph0)
        self.latents[:, :, 0] += torch.randn(2, self.n_trials) * self.R_z

        # print(self.latents.shape)
        for t in range(1, self.dur):
            self.latents[:, :, t] += decay * self.latents[:, :, t - 1]
            self.latents[:, :, t] += (
                V @ self.non_lin(U @ self.latents[:, :, t - 1] + B.unsqueeze(1))
                + torch.randn(2, self.n_trials) * self.R_z
            )

        if task_params["out"] == "rates":
            for t in range(self.dur):
                self.data[:, :, t] = self.non_lin(
                    U @ self.latents[:, :, t] + B.unsqueeze(1)
                )
        elif task_params["out"] == "currents":
            for t in range(self.dur):
                self.data[:, :, t] = U @ self.latents[:, :, t]
        self.data += torch.randn(self.N, self.n_trials, self.dur) * self.R_x
        self.data_eval = self.data

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
            input (torch.tensor; n_inp x self.dur): optional input (here 0)
        """

        return self.data[:, idx], torch.zeros(
            0, self.data.shape[2], device=self.data.device
        )


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

        """
        self.task_params = task_params
        self.dur = task_params["dur"]
        self.n_trials = task_params["n_trials"]
        self.N = task_params["n_neurons"]
        self.w = task_params["w"]
        self.non_lin = task_params["non_lin"]
        self.R_z = task_params["R_z"]

        ph0 = torch.randn(self.n_trials) * np.pi * 2
        if "r0" in task_params:
            r0 = task_params["r0"]
        else:
            r0 = torch.randn(self.n_trials) * 2
        self.latents = torch.zeros(2, self.n_trials, self.dur, dtype=torch.float32)
        self.rates = torch.zeros(self.N, self.n_trials, self.dur, dtype=torch.float32)
        self.latents[0, :, 0] = r0 * np.cos(ph0)
        self.latents[1, :, 0] = r0 * np.sin(ph0)
        self.latents[:, :, 0] += torch.randn(2, self.n_trials) * self.R_z

        for t in range(1, self.dur):
            self.latents[:, :, t] += decay * self.latents[:, :, t - 1]
            self.latents[:, :, t] += (
                V @ self.non_lin(U @ self.latents[:, :, t - 1] + B.unsqueeze(1))
                + torch.randn(2, self.n_trials) * self.R_z
            )

        if task_params["out"] == "rates":
            for t in range(self.dur):
                self.rates[:, :, t] = U @ self.non_lin(
                    self.latents[:, :, t] + B.unsqueeze(1)
                )
        elif task_params["out"] == "currents":
            for t in range(self.dur):
                self.rates[:, :, t] = U @ self.latents[:, :, t]
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
            input (torch.tensor; n_inp x self.dur): optional input (here 0)
        """

        return self.data[:, idx], torch.zeros(0, self.dur, device=self.data.device)


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
        """

        self.dur = task_params["dur"]
        self.n_trials = task_params["n_trials"]
        self.N = task_params["n_neurons"]
        self.w = task_params["w"]
        self.non_lin = task_params["non_lin"]
        self.R_z = task_params["R_z"]
        self.R_x = task_params["R_x"]
        n_repeats = task_params["n_trials"] // task_params_teacher["n_stim"]

        # obtain teacher RNNs stimuli
        reaching = Reaching(task_params_teacher)
        Reaching_loader = DataLoader(
            reaching, batch_size=task_params_teacher["n_stim"], shuffle=False
        )
        s, _, _ = next(iter(Reaching_loader))  # x = trial,time,stims
        s = s.repeat(n_repeats, 1, 1)
        self.stim = s

        # generate teacher data
        ph0 = torch.randn(self.n_trials) * np.pi * 2
        if "r0" in task_params:
            r0 = task_params["r0"]
        else:
            r0 = torch.randn(self.n_trials) * 0.1
        self.latents = torch.zeros(2, self.n_trials, self.dur, dtype=torch.float32)
        self.data = torch.zeros(self.N, self.n_trials, self.dur, dtype=torch.float32)
        self.latents[0, :, 0] = r0 * np.cos(ph0)
        self.latents[1, :, 0] = r0 * np.sin(ph0)
        self.latents[:, :, 0] += torch.randn(2, self.n_trials) * self.R_z

        for t in range(1, self.dur):
            self.latents[:, :, t] = decay * self.latents[:, :, t - 1]
            X = U @ self.latents[:, :, t - 1] + B.unsqueeze(1) + (s[:, t - 1] @ I).T
            self.latents[:, :, t] += (
                V @ self.non_lin(X) + torch.randn(2, self.n_trials) * self.R_z
            )

        if task_params["out"] == "rates":
            for t in range(self.dur):
                self.data[:, :, t] = self.non_lin(
                    U @ self.latents[:, :, t] + B.unsqueeze(1)
                )
        elif task_params["out"] == "currents":
            for t in range(self.dur):
                self.data[:, :, t] = U @ self.latents[:, :, t]
        self.data += torch.randn(self.N, self.n_trials, self.dur) * self.R_x
        self.data_eval = self.data
        self.task_params = task_params

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
        stim = self.stim[idx].T.to(device=self.data.device)
        return self.data[:, idx], stim


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


class NLBDataset(Dataset):
    def __init__(self, data, params, inputs=None):
        self.data = data
        self.data_eval = data  # Not used in this setup
        self.trial_dur = data.shape[0]
        self.task_params = params
        self.stim = (
            inputs.float().to(self.data.device)
            if inputs is not None
            else torch.zeros(
                data.shape[0],
                data.shape[1],
                0,
                device=self.data.device,
                dtype=torch.float,
            )
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx].T, self.stim[idx].T
    

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
        torch.tensor(train_data, dtype=torch.float),
        torch.tensor(eval_data, dtype=torch.float),
        (
            torch.tensor(train_inputs, dtype=torch.float)
            if train_inputs is not None
            else train_inputs
        ),
        (
            torch.tensor(eval_inputs, dtype=torch.float)
            if eval_inputs is not None
            else eval_inputs
        ),
    )