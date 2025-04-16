import torch
import numpy as np

# ADAPTED FROM:
# https://github.com/DurstewitzLab/ChaosRNN
# https://github.com/DurstewitzLab/dendPLRNN
# https://github.com/DurstewitzLab/GTF-shPLRNN


def clean_from_outliers(prior, posterior):
    """Remove outliers (Kls smaller than 0) from the data"""
    mask = (prior > 0) & (posterior > 0)
    return prior[mask], posterior[mask], torch.sum(mask).detach().item()


def kl_between_two_gaussians(mu0, cov0, mu1, cov1):
    """
    For every time step t in mu0 cov0, calculate the kl to all other time steps in mu1, cov1.
    """
    T, n = mu0.shape

    cov1inv_cov0 = torch.einsum("tn,dn->tdn", cov0, 1 / cov1)  # shape T, T, n
    trace_cov1inv_cov0 = torch.sum(cov1inv_cov0, dim=-1)  # shape T,

    diff_mu1_mu0 = mu1.reshape(1, T, n) - mu0.reshape(
        T, 1, n
    )  # subtract every possible combination
    mahalonobis = torch.sum(diff_mu1_mu0 / cov1 * diff_mu1_mu0, dim=2)

    det1 = torch.prod(cov1, dim=1)
    det0 = torch.prod(cov0, dim=1)
    logdiff_det1det0 = torch.log(det1).reshape(1, T) - torch.log(det0).reshape(T, 1)

    kl = 0.5 * (logdiff_det1det0 - n + trace_cov1inv_cov0 + mahalonobis)
    return kl


def eval_likelihood_gmm_for_diagonal_cov(z, mu, std):
    """Evaluate the likelihood of z under a Gaussian mixture model with diagonal covariance matrices"""
    T, dim_x = mu.shape
    S, dim_x = z.shape
    z = z.squeeze().unsqueeze(0)
    precision = 1 / (std ** 2)
    mu = mu.unsqueeze(1)
    vec = z - mu
    exponent = torch.einsum("TSX,TSX->TS", vec, vec)
    exponent *= precision
    sqrt_det_of_cov = std ** dim_x
    likelihood = torch.exp(-0.5 * exponent) / sqrt_det_of_cov
    return likelihood.sum(dim=0) / T


def calc_kl_mc(mu_inf, mu_gen, scale):
    """Calculate the KL divergence between two Gaussian mixture models with diagonal covariance matrices via Monte Carlo sampling"""

    # number of MC samples
    mc_n = 1000
    t = torch.randint(0, mu_inf.shape[0], (mc_n,))

    Norm = torch.randn(mu_inf[t].shape).to(device=mu_inf.device)
    z_sample = mu_inf[t] + scale * Norm

    prior = eval_likelihood_gmm_for_diagonal_cov(z_sample, mu_gen, scale)
    posterior = eval_likelihood_gmm_for_diagonal_cov(z_sample, mu_inf, scale)
    prior, posterior, outlier_ratio = clean_from_outliers(prior, posterior)

    lpost = torch.log(posterior)
    lprior = torch.log(prior)
    kl_mc = torch.mean(lpost - lprior)

    outlier_ratio = 1 - outlier_ratio / mc_n

    return kl_mc, outlier_ratio


def calc_kl_from_data(mu_gen, data_true):
    """Calculate the KL divergence between two datasets after KDE with Gaussian Kernel"""
    time_steps = min(len(data_true), 10000)
    mu_inf = data_true[:time_steps]
    mu_gen = mu_gen[:time_steps]
    scaling = 1.0  # standard deviation of the Gaussian kernel
    kl_mc, o_r = calc_kl_mc(mu_inf.detach(), mu_gen, scaling)
    print("n outliers: " + str(o_r))
    return kl_mc
