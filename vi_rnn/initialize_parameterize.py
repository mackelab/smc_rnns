import torch
import torch.nn as nn
import numpy as np





def chol_cov_embed(x):
    """
    Positive semi-definite embedding of a vector as a lower triangular matrix
    """
    chol_cov = torch.tril(x, diagonal=-1) + torch.diag_embed(
    torch.exp(x[range(x.shape[0]), range(x.shape[0])] / 2))
    return chol_cov

def inverse_chol_cov_embed(x):
    """
    such that x = chol_cov_embed(inverse_chol_cov_embed(x))
    for x a lower triangular matrix
    """
    return torch.diag_embed(torch.log(x[range(x.shape[0]), range(x.shape[0])]))*2 + torch.tril(x, diagonal=-1)




def full_cov_embed(x):
    """
    Return positive semi-definite matrix
    """
    
    cov = lambda x:chol_cov_embed(x) @ (
    chol_cov_embed(x).T
    )
    return cov



def init_noise(noise_type, dim, init_scale, train_noise):
    """
    Initialise noise matrices
    Args:
        noise_type (str): type of noise matrix to use (Full, Diag, Scalar)
        dim (int): length/width of the noise matrix
        init_scale (float): initial scale of the noise (standard deviation)
        train_noise (bool): whether to train the noise matrix
    Returns:
        R (nn.Parameter): noise matrix
        std_embed (function): function to embed the noise matrix as (diagonalised) standard deviation
        var_embed (function): function to embed the noise matrix as covariance
    """
    if noise_type == "full":
        R = nn.Parameter(
        torch.eye(dim) * np.log(init_scale) * 2,
        requires_grad=train_noise,
        )
        std_embed = lambda x: torch.sqrt(
            torch.diagonal(full_cov_embed(x))
        )
        var_embed = lambda x: (full_cov_embed(x))
    elif noise_type == "diag":
        R = nn.Parameter(
        torch.ones(dim) * np.log(init_scale) * 2,
        requires_grad=train_noise,
        )
        std_embed = lambda log_var: torch.exp(log_var / 2)
        var_embed = lambda log_var: torch.exp(log_var)
    elif noise_type == "scalar":
        R = nn.Parameter(
        torch.ones(1) * np.log(init_scale) * 2,
        requires_grad=train_noise,
        )
        std_embed = lambda log_var: torch.exp(log_var / 2).expand(dim)
        var_embed = lambda log_var: torch.exp(log_var).expand(dim)
    return R, std_embed, var_embed




        
def init_AW(dz):
    """Talathi & Vartak 2016: Improving Performance of Recurrent Neural Network with ReLU Nonlinearity
    code  adapted from https://github.com/DurstewitzLab/dendPLRNN"""
    matrix_random = torch.randn(dz, dz)
    matrix_positive_normal = (1 / (dz)) * matrix_random.T @ matrix_random
    matrix = torch.eye(dz) + matrix_positive_normal
    max_ev = torch.max(abs(torch.linalg.eigvals(matrix)))
    matrix_spectral_norm_one = matrix / max_ev
    A = matrix_spectral_norm_one[range(dz), range(dz)]
    return nn.Parameter(A, requires_grad=True)


def init_AW_exp_par(dz):
    """exp param of Talathi & Vartak 2016: Improving Performance of Recurrent Neural Network with ReLU Nonlinearity
    code apapted from: https://github.com/DurstewitzLab/dendPLRNN"""
    matrix_random = torch.randn(dz, dz)
    matrix_positive_normal = 1 / (dz * dz) * matrix_random @ matrix_random.T
    matrix = torch.eye(dz) + matrix_positive_normal
    max_ev = torch.max(abs(torch.linalg.eigvals(matrix)))
    matrix_spectral_norm_one = matrix / max_ev
    A = torch.log(-torch.log(matrix_spectral_norm_one[range(dz), range(dz)]))
    return nn.Parameter(A, requires_grad=True)


def initialize_Ws_uniform(dz, N):
    """Initialize the weights of the network
    Args:
        dz (int): dimensionality of the latent space
        N (int): dimensionality of the data
    Returns:
        n (nn.Parameter): right singular vecs,  Uniform between -1/sqrt(N) and 1/sqrt(N)
        m (nn.Parameter): left singular vecs,  Uniform between -1/sqrt(dz) and 1/sqrt(dz)
    """
    print("using uniform init")
    n = uniform_init2d(dz, N)
    m = uniform_init2d(N, dz)
    return nn.Parameter(n, requires_grad=True), nn.Parameter(m, requires_grad=True)


def initialize_Ws_gauss(dz, N, scaling):
    """Initialize the weights of the network with (correlated) Gaussians
    Args:
        dz (int): dimensionality of the latent space
        N (int): dimensionality of the data
        scaling (float): scaling factor
    Returns:
        n (nn.Parameter): right singular vecs, with sd 1/(scaling*sqrt(3 N))
        m (nn.Parameter): left singular vecs, with sd 1/sqrt(3 dz)
    """
    print("using gauss init")
    cov = torch.eye(dz * 2)
    for i in range(dz):
        cov[i, dz + i] = 0.6
        cov[dz + i, i] = 0.6
    chol_cov = torch.linalg.cholesky(cov)
    loadings = chol_cov @ torch.randn(dz * 2, N)
    n = loadings[:dz, :] / (scaling * np.sqrt(3 * N))
    m = loadings[dz:, :] / np.sqrt(3 * dz)
    return nn.Parameter(n, requires_grad=True), nn.Parameter(m.T, requires_grad=True)


def uniform_init2d(dim1, dim2):
    """Uniform init between -1/sqrt(dim2) and 1/sqrt(dim2)"""
    r = 1 / np.sqrt(dim2)
    return (r * 2 * torch.rand(dim1, dim2)) - r


def uniform_init1d(dim1):
    """Uniform init between -1/sqrt(dim1) and 1/sqrt(dim1)"""
    r = 1 / np.sqrt(dim1)
    return (r * 2 * torch.rand(dim1)) - r


def exp_par_F(A):
    """Exponential parameterisation of the time constants"""
    return torch.exp(-torch.exp(A)).unsqueeze(0).unsqueeze(2).unsqueeze(3)


def diag_mid(x):
    """make BXXT matrix from BXT by diagonalising."""
    return torch.diag_embed(x.permute(0, 2, 1)).permute(0, 2, 3, 1)


def clamp_norm(w, max_norm=1, dim=0):
    """Clamp the norm of the weights to max_norm"""
    with torch.no_grad():
        norm = w.norm(2, dim=dim, keepdim=True).clamp(min=max_norm / 2)
        desired = torch.clamp(norm, max=max_norm)
        w *= desired / norm
    return w


def split_diag_offdiag(A):
    """Split a matrix into diagonal and off-diagonal parts"""
    diag = torch.diag(torch.diag(A))
    off_diag = A - diag
    return diag, off_diag


def drelu_dx(x):
    """Derivative of ReLU"""
    return torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))


def relu_derivative(x, h):
    """Deritive of ReLU at x-h"""
    return drelu_dx(x - h)


def clipped_relu_derivative(x, h):
    """Derivative of clipped ReLU at x-h"""
    return drelu_dx(x + h) - drelu_dx(x)


def tanh_derivative(x, h):
    """Derivative of tanh at x-h"""
    return 1 - torch.tanh(x - h) ** 2


def gram_schmidt(X):
    """Gram-Schmidt orthogonalization"""
    orth_basis = []
    orth_basis.append(X[:, 0])
    for i in range(1, X.shape[1]):
        substr = torch.zeros_like(X[:, i])
        for vect in orth_basis:
            overlap = torch.inner(X[:, i], vect) / torch.inner(vect, vect)
            substr += overlap * vect
        orth_basis.append(X[:, i] - substr)
    Y = torch.stack(orth_basis)
    return Y.T


def orth_proj(m, x):
    """Orthogonal projection of x onto m
    Args:
        m (torch.Tensor): vector to be projected onto, NxZ
        x (torch.Tensor): vector to be projected, BxN
    Returns:
        z (torch.Tensor): projection, BxZ

    """
    projection_matrix = torch.linalg.inv(m.T @ m) @ m.T
    z = (projection_matrix @ x.T).T
    return z


class normalize_m:
    """Normalize a matrix by its norm"""

    def __init__(self, m):
        """Initialize the normalizer"""
        self.init_norm = m
        self.norm = torch.norm(m, dim=0, keepdim=True)

    def __call__(self, X):
        """Normalize and orthogonalise the matrix"""
        orth_basis = []
        orth_basis.append(X[:, 0])

        for i in range(1, X.shape[1]):
            substr = torch.zeros_like(X[:, i])
            for vect in orth_basis:
                overlap = torch.inner(X[:, i], vect) / torch.inner(vect, vect)
                substr += overlap * vect
            orth_basis.append(X[:, i] - substr)
        Y = torch.stack(orth_basis)
        Y = Y / torch.norm(Y, dim=1, keepdim=True) * self.norm.T
        return Y.T
