import pickle
import torch
from types import SimpleNamespace
from typing import Union, Optional
from pathlib import Path

from .lfads_modules.decoder import Decoder
from .lfads_modules.priors import MultivariateNormal, AutoregressiveMultivariateNormal
from .lfads_modules.readout import FanInLinear


class LFADSAutonomousSampler:
    def __init__(
        self,
        decoder: Decoder,
        ic_prior: MultivariateNormal,
        readout: FanInLinear,
        behavior_readout: Optional[torch.nn.Linear] = None,
        device: Union[str, torch.device] = "cpu",
        seed: Optional[int] = None,
    ):
        super().__init__()
        assert (
            not decoder.rnn.cell.use_con
        ), f"LFADS generator with controller not supported"
        self.device = device
        self.decoder = decoder.to(device)
        self.decoder.eval()
        self.ic_prior = ic_prior.to(device)
        self.ic_prior.eval()
        self.readout = readout.to(device)
        self.readout.eval()
        if behavior_readout is not None:
            self.behavior_readout = behavior_readout.to(device)
            self.behavior_readout.eval()
        else:
            self.behavior_readout = None
        self.generator = torch.Generator(device=device)
        if seed is not None:
            self.generator.manual_seed(seed)

    def set_seed(self, seed: int = None):
        if seed is not None:
            self.generator.manual_seed(seed)

    @torch.no_grad()
    def sample_prior(self, n: int):
        mean = self.ic_prior.mean.data
        std = torch.exp(0.5 * self.ic_prior.logvar)
        return torch.normal(
            mean.view(1, -1).expand(n, -1),
            std.view(1, -1).expand(n, -1),
            generator=self.generator,
        )

    @torch.no_grad()
    def sample_latents(
        self,
        n: int,
        t: int,
        ext_input: Optional[torch.Tensor] = None,
    ):
        ic_samp = self.sample_prior(n)
        ci = torch.zeros(n, t, 0, device=self.device)
        if ext_input is None:
            ext_input = torch.zeros(
                n, t, self.decoder.hparams.ext_input_dim, device=self.device
            )
        self.decoder.hparams.recon_seq_len = t
        factors = self.decoder(ic_samp, ci, ext_input)[-1]
        return factors

    @torch.no_grad()
    def sample_observations(
        self,
        n: int,
        t: int,
        ext_input: Optional[torch.Tensor] = None,
        return_rates: bool = False,
    ):
        factors = self.sample_latents(n=n, t=t, ext_input=ext_input)
        rates = torch.exp(self.readout(factors))
        spikes = torch.poisson(rates, generator=self.generator)
        if return_rates:
            return spikes, rates
        return spikes

    @torch.no_grad()
    def sample_behavior(
        self,
        n: int,
        t: int,
        ext_input: Optional[torch.Tensor] = None,
    ):
        if self.behavior_readout is None:
            raise AttributeError(
                f"`LFADSUnconditionalSampler.behavior_readout` is not assigned. "
            )
        factors = self.sample_latents(n=n, t=t, ext_input=ext_input)
        rates = torch.exp(self.readout(factors))
        behavior = self.behavior_readout(rates)
        return behavior

    @torch.no_grad()
    def sample_everything(
        self,
        n: int,
        t: int,
        ext_input: Optional[torch.Tensor] = None,
        include_behavior: bool = False,
    ):
        factors = self.sample_latents(n=n, t=t, ext_input=ext_input)
        rates = torch.exp(self.readout(factors))
        spikes = torch.poisson(rates, generator=self.generator)
        returns = (spikes, rates, factors)
        if include_behavior and self.behavior_readout is None:
            raise AttributeError(
                f"`{self.__class__.__name__}.behavior_readout` is not assigned. "
            )
        elif include_behavior:
            behavior = self.behavior_readout(rates)
            returns += (behavior,)
        return returns


class LFADSControllerSampler:
    def __init__(
        self,
        decoder: Decoder,
        ic_prior: MultivariateNormal,
        readout: FanInLinear,
        co_prior: Optional[AutoregressiveMultivariateNormal] = None,
        behavior_readout: Optional[torch.nn.Linear] = None,
        device: Union[str, torch.device] = "cpu",
        seed: Optional[int] = None,
    ):
        super().__init__()
        assert decoder.prior_sampling, f"Decoder must be in prior sampling mode"
        self.device = device
        self.decoder = decoder.to(device)
        self.decoder.eval()
        self.ic_prior = ic_prior.to(device)
        self.ic_prior.eval()
        self.readout = readout.to(device)
        self.readout.eval()
        self.co_prior = co_prior.to(device)
        self.co_prior.eval()
        if behavior_readout is not None:
            self.behavior_readout = behavior_readout.to(device)
            self.behavior_readout.eval()
        else:
            self.behavior_readout = None
        self.generator = torch.Generator(device=device)
        if seed is not None:
            self.generator.manual_seed(seed)

    def set_seed(self, seed: int = None):
        if seed is not None:
            self.generator.manual_seed(seed)

    @torch.no_grad()
    def sample_prior(self, n: int):
        mean = self.ic_prior.mean.data
        std = torch.exp(0.5 * self.ic_prior.logvar)
        return torch.normal(
            mean.view(1, -1).expand(n, -1),
            std.view(1, -1).expand(n, -1),
            generator=self.generator,
        )

    @torch.no_grad()
    def sample_controller(self, n: int, t: int):
        alphas = torch.exp(-1.0 / torch.exp(self.co_prior.logtaus))
        init_mean = torch.zeros(n, self.co_prior.logtaus.shape[0])
        init_std = torch.exp(0.5 * (self.co_prior.lognvars - torch.log(1 - alphas**2)))
        samples = [
            torch.normal(
                init_mean,
                init_std.view(1, -1).expand(n, -1),
                generator=self.generator,
            )
        ]
        std = torch.exp(0.5 * self.co_prior.lognvars)
        for i in range(t - 1):
            mean = samples[-1] * alphas[None, :]
            samples.append(
                torch.normal(
                    mean,
                    std.view(1, -1).expand(n, -1),
                    generator=self.generator,
                )
            )
        return torch.stack(samples, dim=1)

    @torch.no_grad()
    def sample_latents(
        self,
        n: int,
        t: int,
        ext_input: Optional[torch.Tensor] = None,
    ):
        ic_samp = self.sample_prior(n)
        ci = self.sample_controller(n, t)
        if ext_input is None:
            ext_input = torch.zeros(
                n, t, self.decoder.hparams.ext_input_dim, device=self.device
            )
        self.decoder.hparams.recon_seq_len = t
        factors = self.decoder(ic_samp, ci, ext_input)[-1]
        return factors

    @torch.no_grad()
    def sample_observations(
        self,
        n: int,
        t: int,
        ext_input: Optional[torch.Tensor] = None,
        return_rates: bool = False,
    ):
        factors = self.sample_latents(n=n, t=t, ext_input=ext_input)
        rates = torch.exp(self.readout(factors))
        spikes = torch.poisson(rates, generator=self.generator)
        if return_rates:
            return spikes, rates
        return spikes

    @torch.no_grad()
    def sample_behavior(
        self,
        n: int,
        t: int,
        ext_input: Optional[torch.Tensor] = None,
    ):
        if self.behavior_readout is None:
            raise AttributeError(
                f"`LFADSUnconditionalSampler.behavior_readout` is not assigned. "
            )
        factors = self.sample_latents(n=n, t=t, ext_input=ext_input)
        rates = torch.exp(self.readout(factors))
        behavior = self.behavior_readout(rates)
        return behavior

    @torch.no_grad()
    def sample_everything(
        self,
        n: int,
        t: int,
        ext_input: Optional[torch.Tensor] = None,
        include_behavior: bool = False,
    ):
        factors = self.sample_latents(n=n, t=t, ext_input=ext_input)
        rates = torch.exp(self.readout(factors))
        spikes = torch.poisson(rates, generator=self.generator)
        returns = (spikes, rates, factors)
        if include_behavior and self.behavior_readout is None:
            raise AttributeError(
                f"`{self.__class__.__name__}.behavior_readout` is not assigned. "
            )
        elif include_behavior:
            behavior = self.behavior_readout(rates)
            returns += (behavior,)
        return returns


def load_lfads_sampler(
    file_path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
    seed: Optional[int] = None,
    autonomous=True,
):
    with open(file_path, "rb") as f:
        sampler_state_dict = pickle.load(f)
    # load prior
    ic_prior_state_dict = sampler_state_dict["ic_prior"]
    ic_prior = MultivariateNormal(0.0, 0.1, shape=ic_prior_state_dict["mean"].shape[0])
    ic_prior.load_state_dict(ic_prior_state_dict)
    # load decoder
    decoder_state_dict = sampler_state_dict["decoder"]
    if "rnn.cell.co_linear.weight" in decoder_state_dict:
        co_dim = decoder_state_dict["rnn.cell.co_linear.bias"].shape[0] // 2
        ci_enc_dim = (
            decoder_state_dict["rnn.cell.con_cell.weight_ih"].shape[1]
            - decoder_state_dict["rnn.cell.fac_linear.weight"].shape[0]
        ) // 2
    else:
        co_dim = 0
        ci_enc_dim = 0
    hps = SimpleNamespace(  # hacky solution to not preserving full model config
        dropout_rate=0.0,
        ic_dim=decoder_state_dict["ic_to_g0.weight"].shape[1],
        gen_dim=decoder_state_dict["ic_to_g0.weight"].shape[0],
        con_dim=decoder_state_dict["con_h0"].shape[1],
        recon_seq_len=0,
        ext_input_dim=0,
        co_dim=co_dim,
        cell_clip=5.0,
        fac_dim=decoder_state_dict["rnn.cell.fac_linear.weight"].shape[0],
        ci_enc_dim=ci_enc_dim,
    )
    decoder = Decoder(hps, prior_sampling=(not autonomous))
    decoder.load_state_dict(decoder_state_dict)
    # load readout
    readout_state_dict = sampler_state_dict["readout"]
    readout = FanInLinear(
        readout_state_dict["weight"].shape[1], readout_state_dict["weight"].shape[0]
    )
    readout.load_state_dict(readout_state_dict)
    # load behavior readout
    if "behavior_readout" in sampler_state_dict:
        behavior_readout_state_dict = sampler_state_dict["behavior_readout"]
        behavior_readout = torch.nn.Linear(
            behavior_readout_state_dict["weight"].shape[1],
            behavior_readout_state_dict["weight"].shape[0],
        )
        behavior_readout.load_state_dict(behavior_readout_state_dict)
    else:
        behavior_readout = None
    # create sampler
    if autonomous:
        sampler = LFADSAutonomousSampler(
            decoder=decoder,
            ic_prior=ic_prior,
            readout=readout,
            behavior_readout=behavior_readout,
            device=device,
            seed=seed,
        )
    else:
        co_prior_state_dict = sampler_state_dict["co_prior"]
        co_prior = AutoregressiveMultivariateNormal(
            0.0, 0.1, shape=co_prior_state_dict["lognvars"].shape[0]
        )
        co_prior.load_state_dict(co_prior_state_dict)

        sampler = LFADSControllerSampler(
            decoder=decoder,
            ic_prior=ic_prior,
            co_prior=co_prior,
            readout=readout,
            behavior_readout=behavior_readout,
            device=device,
            seed=seed,
        )
    return sampler


def save_lfads_sampler(sampler, file_path: Union[str, Path]):
    sampler_state_dict = dict(
        ic_prior=sampler.ic_prior.state_dict(),
        decoder=sampler.decoder.state_dict(),
        readout=sampler.readout.state_dict(),
    )
    if sampler.behavior_readout is not None:
        sampler_state_dict["behavior_readout"] = (
            sampler.behavior_readout.state_dict(),
        )
    if hasattr(sampler, "co_prior"):
        sampler_state_dict["co_prior"] = sampler.co_prior.state_dict()
    with open(file_path, "wb") as f:
        pickle.dump(sampler_state_dict, f)
