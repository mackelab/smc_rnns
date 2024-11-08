"""
lfads-torch modules needed for generator

Author: Andrew Sedler
Sources: 
  - https://github.com/arsedler9/lfads-torch/blob/d8b4b3ba87a49fd74a3c06afb3ec1b695c6a2227/lfads_torch/modules/recurrent.py
  - https://github.com/arsedler9/lfads-torch/blob/d8b4b3ba87a49fd74a3c06afb3ec1b695c6a2227/lfads_torch/modules/decoder.py
"""

import torch
import torch.nn.functional as F
from torch import nn


class ClippedGRUCell(nn.GRUCell):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        clip_value: float = float("inf"),
        is_encoder: bool = False,
    ):
        super().__init__(input_size, hidden_size, bias=True)
        self.bias_hh.requires_grad = False
        self.clip_value = clip_value
        # scale_dim = input_size + hidden_size if is_encoder else None  # NOTE: removed, unnecessary
        # init_gru_cell_(self, scale_dim=scale_dim)  # NOTE: removed, unnecessary

    def forward(self, input: torch.Tensor, hidden: torch.Tensor):
        x_all = input @ self.weight_ih.T + self.bias_ih
        x_z, x_r, x_n = torch.chunk(x_all, chunks=3, dim=1)
        split_dims = [2 * self.hidden_size, self.hidden_size]
        weight_hh_zr, weight_hh_n = torch.split(self.weight_hh, split_dims)
        bias_hh_zr, bias_hh_n = torch.split(self.bias_hh, split_dims)
        h_all = hidden @ weight_hh_zr.T + bias_hh_zr
        h_z, h_r = torch.chunk(h_all, chunks=2, dim=1)
        z = torch.sigmoid(x_z + h_z)
        r = torch.sigmoid(x_r + h_r)
        h_n = (r * hidden) @ weight_hh_n.T + bias_hh_n
        n = torch.tanh(x_n + h_n)
        hidden = z * hidden + (1 - z) * n
        hidden = torch.clamp(hidden, -self.clip_value, self.clip_value)
        return hidden


class KernelNormalizedLinear(nn.Linear):
    def forward(self, input):
        normed_weight = F.normalize(self.weight, p=2, dim=1)
        return F.linear(input, normed_weight, self.bias)


class DecoderCell(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hps = hparams
        # Create the generator
        self.gen_cell = ClippedGRUCell(
            hps.ext_input_dim + hps.co_dim, hps.gen_dim, clip_value=hps.cell_clip
        )
        # Create the mapping from generator states to factors
        self.fac_linear = KernelNormalizedLinear(hps.gen_dim, hps.fac_dim, bias=False)
        # init_linear_(self.fac_linear)  # NOTE: removed, unnecessary
        # Create the dropout layer
        self.dropout = nn.Dropout(hps.dropout_rate)
        # Decide whether to use the controller
        self.use_con = all(
            [
                hps.ci_enc_dim > 0,
                hps.con_dim > 0,
                hps.co_dim > 0,
            ]
        )
        if self.use_con:
            # Create the controller
            self.con_cell = ClippedGRUCell(
                2 * hps.ci_enc_dim + hps.fac_dim, hps.con_dim, clip_value=hps.cell_clip
            )
            # Define the mapping from controller state to controller output parameters
            self.co_linear = nn.Linear(hps.con_dim, hps.co_dim * 2)
            # init_linear_(self.co_linear)  # NOTE: removed, unnecessary
        # Keep track of the state dimensions
        self.state_dims = [
            hps.gen_dim,
            hps.con_dim,
            hps.co_dim,
            hps.co_dim,
            hps.co_dim + hps.ext_input_dim,
            hps.fac_dim,
        ]
        # Keep track of the input dimensions
        self.input_dims = [2 * hps.ci_enc_dim, hps.ext_input_dim]

    def forward(self, input, h_0, sample_posteriors=True):
        hps = self.hparams

        # Split the state up into variables of interest
        gen_state, con_state, co_mean, co_std, gen_input, factor = torch.split(
            h_0, self.state_dims, dim=1
        )
        ci_step, ext_input_step = torch.split(input, self.input_dims, dim=1)

        if self.use_con:
            # Compute controller inputs with dropout
            con_input = torch.cat([ci_step, factor], dim=1)
            con_input_drop = self.dropout(con_input)
            # Compute and store the next hidden state of the controller
            con_state = self.con_cell(con_input_drop, con_state)
            # Compute the distribution of the controller outputs at this timestep
            co_params = self.co_linear(con_state)
            co_mean, co_logvar = torch.split(co_params, hps.co_dim, dim=1)
            co_std = torch.sqrt(torch.exp(co_logvar))
            # Sample from the distribution of controller outputs
            co_post = self.hparams.co_prior.make_posterior(co_mean, co_std)
            con_output = co_post.rsample() if sample_posteriors else co_mean
            # Combine controller output with any external inputs
            gen_input = torch.cat([con_output, ext_input_step], dim=1)
        else:
            # If no controller is being used, can still provide ext inputs
            gen_input = ext_input_step
        # compute and store the next
        gen_state = self.gen_cell(gen_input, gen_state)
        gen_state_drop = self.dropout(gen_state)
        factor = self.fac_linear(gen_state_drop)

        hidden = torch.cat(
            [gen_state, con_state, co_mean, co_std, gen_input, factor], dim=1
        )

        return hidden


class PriorDecoderCell(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hps = hparams
        # Create the generator
        self.gen_cell = ClippedGRUCell(
            hps.ext_input_dim + hps.co_dim, hps.gen_dim, clip_value=hps.cell_clip
        )
        # Create the mapping from generator states to factors
        self.fac_linear = KernelNormalizedLinear(hps.gen_dim, hps.fac_dim, bias=False)
        # init_linear_(self.fac_linear)  # NOTE: removed, unnecessary
        # Create the dropout layer
        self.dropout = nn.Dropout(hps.dropout_rate)
        # Decide whether to use the controller
        self.use_con = all(
            [
                hps.ci_enc_dim > 0,
                hps.con_dim > 0,
                hps.co_dim > 0,
            ]
        )
        if self.use_con:
            # Create the controller
            self.con_cell = ClippedGRUCell(
                2 * hps.ci_enc_dim + hps.fac_dim, hps.con_dim, clip_value=hps.cell_clip
            )
            # Define the mapping from controller state to controller output parameters
            self.co_linear = nn.Linear(hps.con_dim, hps.co_dim * 2)
            # init_linear_(self.co_linear)  # NOTE: removed, unnecessary
        # Keep track of the state dimensions
        self.state_dims = [
            hps.gen_dim,
            hps.con_dim,
            hps.co_dim,
            hps.co_dim,
            hps.co_dim + hps.ext_input_dim,
            hps.fac_dim,
        ]
        # Keep track of the input dimensions
        self.input_dims = [hps.co_dim, hps.ext_input_dim]

    def forward(self, input, h_0, sample_posteriors=True):
        hps = self.hparams

        # Split the state up into variables of interest
        gen_state, con_state, co_mean, co_std, gen_input, factor = torch.split(
            h_0, self.state_dims, dim=1
        )
        ci_step, ext_input_step = torch.split(input, self.input_dims, dim=1)

        if self.use_con:
            con_output = ci_step  # "ci" is sampled controller output
            # Combine controller output with any external inputs
            gen_input = torch.cat([con_output, ext_input_step], dim=1)
        else:
            # If no controller is being used, can still provide ext inputs
            gen_input = ext_input_step
        # compute and store the next
        gen_state = self.gen_cell(gen_input, gen_state)
        gen_state_drop = self.dropout(gen_state)
        factor = self.fac_linear(gen_state_drop)

        hidden = torch.cat(
            [gen_state, con_state, co_mean, co_std, gen_input, factor], dim=1
        )

        return hidden


class DecoderRNN(nn.Module):
    def __init__(self, hparams, prior_sampling=False):
        super().__init__()
        if prior_sampling:
            self.cell = PriorDecoderCell(hparams=hparams)
        else:
            self.cell = DecoderCell(hparams=hparams)

    def forward(self, input, h_0, sample_posteriors=True):
        hidden = h_0
        input = torch.transpose(input, 0, 1)
        output = []
        for input_step in input:
            hidden = self.cell(input_step, hidden, sample_posteriors=sample_posteriors)
            output.append(hidden)
        output = torch.stack(output, dim=1)
        return output, hidden


class Decoder(nn.Module):
    def __init__(self, hparams, prior_sampling=False):
        super().__init__()
        self.prior_sampling = prior_sampling
        self.hparams = hps = hparams

        self.dropout = nn.Dropout(hps.dropout_rate)
        # Create the mapping from ICs to gen_state
        self.ic_to_g0 = nn.Linear(hps.ic_dim, hps.gen_dim)
        # init_linear_(self.ic_to_g0)  # NOTE: removed, unnecessary
        # Create the decoder RNN
        self.rnn = DecoderRNN(hparams=hparams, prior_sampling=prior_sampling)
        # Initial hidden state for controller
        self.con_h0 = nn.Parameter(torch.zeros((1, hps.con_dim), requires_grad=True))

    def forward(self, ic_samp, ci, ext_input, sample_posteriors=True):
        hps = self.hparams

        # Get size of current batch (may be different than hps.batch_size)
        batch_size = ic_samp.shape[0]
        # Calculate initial generator state and pass it to the RNN with dropout rate
        gen_init = self.ic_to_g0(ic_samp)
        gen_init_drop = self.dropout(gen_init)
        # Pad external inputs if necessary and perform dropout
        fwd_steps = hps.recon_seq_len - ext_input.shape[1]
        if fwd_steps > 0:
            pad = torch.zeros(batch_size, fwd_steps, hps.ext_input_dim)
            ext_input = torch.cat([ext_input, pad.to(ext_input.device)], axis=1)
        ext_input_drop = self.dropout(ext_input)
        # Prepare the decoder inputs and and initial state of decoder RNN
        dec_rnn_input = torch.cat([ci, ext_input_drop], dim=2)
        device = gen_init.device
        dec_rnn_h0 = torch.cat(
            [
                gen_init,
                torch.tile(self.con_h0, (batch_size, 1)),
                torch.zeros((batch_size, hps.co_dim), device=device),
                torch.ones((batch_size, hps.co_dim), device=device),
                torch.zeros(
                    (batch_size, hps.co_dim + hps.ext_input_dim), device=device
                ),
                self.rnn.cell.fac_linear(gen_init_drop),
            ],
            dim=1,
        )
        states, _ = self.rnn(
            dec_rnn_input, dec_rnn_h0, sample_posteriors=sample_posteriors
        )
        split_states = torch.split(states, self.rnn.cell.state_dims, dim=2)
        dec_output = (gen_init, *split_states)

        return dec_output
