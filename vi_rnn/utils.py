import torch
import numpy as np
from torch.utils.data import Dataset
from initialize_parameterize import chol_cov_embed, inverse_chol_cov_embed


def np_relu(x):
    """ReLU function for numpy"""
    return np.maximum(x, 0)


def extract_phase_plane_rnn(rnn, xlims, ylims, n_points=30, inp=None):
    """Extract the phase plane of the RNN.
    Args:
        rnn: RNN, RNN model
        xlims: float or list, limits for x-axis
        ylims: float or list, limits for y-axis
        n_points: int, number of points to sample
        inp: np.array (n_inp), input to the RNN
    Returns:
        X: np.array (n_points x n_points), x-axis meshgrid
        Y: np.array (n_points x n_points), y-axis meshgrid
        u: np.array (n_points x n_points), x-axis velocity field
        v: np.array (n_points x n_points), y-axis velocity field
        norm: np.array (n_points x n_points), norm of the velocity field


    """
    U, V, B = extract_orth_basis_rnn(rnn)
    U = U.detach().numpy()
    V = V.detach().numpy()
    B = B.detach().numpy()
    if inp is None:
        I = np.zeros(1)
        inp = np.zeros(1)
    else:
        I = rnn.rnn.w_inp.detach().numpy()
    alpha = rnn.rnn.dt / rnn.rnn.tau

    if not isinstance(xlims, list):
        xlims = [-xlims, xlims]
    if not isinstance(ylims, list):
        ylims = [-ylims, ylims]

    def dyn_eq(x, y):
        z = np.array([x, y])
        dz = -z + V @ np_relu(U @ z + B + inp @ I) / alpha
        dz /= rnn.rnn.tau
        return dz[0], dz[1]

    X, Y = np.meshgrid(
        np.linspace(xlims[0], xlims[1], n_points),
        np.linspace(ylims[0], ylims[1], n_points),
    )
    u, v = np.zeros_like(X), np.zeros_like(X)
    NI, NJ = X.shape

    norm = np.zeros((NI, NJ))
    for i in range(NI):
        for j in range(NJ):
            x, y = X[i, j], Y[i, j]
            dx, dy = dyn_eq(x, y)
            u[i, j] = dx
            v[i, j] = dy
            norm[i, j] = np.log(np.linalg.norm([dx, dy]))
    return X, Y, u, v, norm


def extract_phase_plane_vae(vae, xlims, ylims, n_points=30, h=10, inp=None):
    """Extract the phase plane of the vae.rnn.
    Args:
        vae: VAE, VAE model
        xlims: float, limits for x-axis
        ylims: float, limits for y-axis
        n_points: int, number of points to sample
        h: float, step size (in ms)
    Returns:
        X: np.array (n_points x n_points), x-axis meshgrid
        Y: np.array (n_points x n_points), y-axis meshgrid
        u: np.array (n_points x n_points), x-axis velocity field
        v: np.array (n_points x n_points), y-axis velocity field
        norm: np.array (n_points x n_points), norm of the velocity field"""
    prior = vae.rnn.transition
    decay = prior.cast_decay(prior.decay).detach().numpy().squeeze()
    V = prior.n.detach().numpy()
    U = prior.m.detach().numpy()
    B = prior.h.detach().numpy()
    if inp is None:
        I = np.zeros(1)
        inp = np.zeros(1)
    else:
        I = prior.Wu.detach().numpy()

    if not isinstance(xlims, list):
        xlims = [-xlims, xlims]
    if not isinstance(ylims, list):
        ylims = [-ylims, ylims]

    def dyn_eq(x, y):
        z = np.array([x, y])
        zn = (decay) * z + V @ np_relu(U @ z - B + I @ inp)
        dz = (zn - z) / h
        return dz[0], dz[1]

    X, Y = np.meshgrid(
        np.linspace(xlims[0], xlims[1], n_points),
        np.linspace(ylims[0], ylims[1], n_points),
    )
    u, v = np.zeros_like(X), np.zeros_like(X)
    NI, NJ = X.shape

    norm = np.zeros((NI, NJ))
    for i in range(NI):
        for j in range(NJ):
            x, y = X[i, j], Y[i, j]
            dx, dy = dyn_eq(x, y)
            u[i, j] = dx
            v[i, j] = dy
            norm[i, j] = np.log(np.linalg.norm([dx, dy]))
    return X, Y, u, v, norm


def orthogonalise_network(vae):
    """
    Orthogonalise loadings of the LR RNN prior

    Warning: at the moment this makes the network unable
    to train afterwards as the cholesky decomposition is not constrained

    Additionally the output mapping might need to be adjusted

    Args:
        vae (VAE): VAE model
    Returns:
        vae (VAE): VAE model with orthogonalised loadings

    """
    with torch.no_grad():
        m_or = vae.rnn.transition.m  # 20,2
        n_or = vae.rnn.transition.n
        J = m_or @ n_or
        u, s, v = torch.linalg.svd(J)
        projection_matrix = u[:, : vae.dim_z].T @ m_or

        if vae.rnn.params["noise_z"] == "full":
            proj_chol = chol_cov_embed(vae.rnn.R_z)
        else:
            proj_chol = torch.diag(vae.rnn.std_embed_z(vae.rnn.R_z))

        proj_chol = projection_matrix @ proj_chol

        m_new = u[:, : vae.dim_z]
        n_new = (v[: vae.dim_z].T * s[: vae.dim_z]).T
        vae.rnn.transition.m.copy_(m_new)
        vae.rnn.transition.n.copy_(n_new)
        vae.rnn.R_z = torch.nn.Parameter(
            inverse_chol_cov_embed(torch.linalg.cholesky(proj_chol @ proj_chol.T))
        )
        vae.rnn.params["noise_z"] = "full"
    return vae


def extract_orth_basis_rnn(rnn):
    """
    Extract orthogonal basis of the RNN
    Args:
        rnn: RNN, RNN model
    Returns:
        U: torch.Tensor (N,tr), left singular vectors
        V: torch.Tensor (tr,N), (scaled) right singular vectors
        B: torch.Tensor (N,), biases

    """
    U = torch.clone(rnn.rnn.m.detach())
    N, tr = U.shape
    alpha = rnn.rnn.dt / rnn.rnn.tau
    V = torch.clone(rnn.rnn.n.detach() * alpha / N)
    W_or = U @ V
    U, s, V = torch.linalg.svd(W_or, full_matrices=False)
    U, s, V = U[:, :tr], s[:tr], V[:tr, :]
    V = (V.T * s).T
    if torch.linalg.norm(W_or - U @ V) > 1e-5:
        print("Warning: not orthogonal")
        print(torch.linalg.norm(W_or - U @ V))
    B = rnn.rnn.b_rec.detach()
    return U, V, B


def rotate_basis_vectors(vae, rotation):
    """
    Rotate the basis vectors of the vae.rnn
    Args:
        vae: VAE, VAE model
        rotation: torch.Tensor (dim_z,dim_z), transformation matrix
    Returns:
        vae: VAE, VAE model with rotated basis vectors
    """

    with torch.no_grad():
        m_or = vae.rnn.transition.m  # 20,2
        n_or = vae.rnn.transition.n
        m_new = m_or @ np.linalg.inv(rotation)
        n_new = rotation @ n_or
        if vae.rnn.params["noise_z"] == "full":
            proj_chol = chol_cov_embed(vae.rnn.R_z)
        else:
            proj_chol = torch.diag(vae.rnn.std_embed_z(vae.rnn.R_z))
        proj_chol = rotation @ proj_chol
        vae.rnn.transition.m.copy_(m_new)
        vae.rnn.transition.n.copy_(n_new)
        chol_cov_embed = lambda x: torch.tril(x)
        vae.rnn.R_z = torch.nn.Parameter(
            inverse_chol_cov_embed(torch.linalg.cholesky(proj_chol @ proj_chol.T))
        )
        vae.rnn.params["noise_z"] = "full"
    return vae


def get_loadings(vae):
    """
    Extract the loadings of the vae.rnn
    Args:
        vae: VAE, VAE model
    Returns:
        tau: np.array (dim_z,), time constants
        pV: np.array (dim_z,dim_x), right singular vectors
        pU: np.array (dim_x,dim_z), (scaled) left singular vectors
        pB: np.array (dim_x,), biases
        pI: np.array (dim_x,), input weights

    """
    prior = vae.rnn.transition
    tau = prior.cast_decay(prior.decay).detach().numpy().squeeze()
    pV = prior.n.detach().numpy()
    pU = prior.m.detach().numpy()
    pB = prior.h.detach().numpy()
    pI = prior.Wu.detach().numpy()
    return tau, pV, pU, pB, pI


def get_orth_proj_latents(vae):
    """
    For projection latents on orthogonalised basis
    """
    with torch.no_grad():
        m_or = vae.rnn.transition.m
        n_or = vae.rnn.transition.n
        J = m_or @ n_or
        u, s, v = torch.linalg.svd(J)
        projection_matrix = u[:, : vae.dim_z].T @ m_or
    return projection_matrix
