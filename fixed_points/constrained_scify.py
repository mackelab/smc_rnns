# Code adapted from:
# https://github.com/DurstewitzLab/CNS-2023
# See: Eisenmann L., Monfared M., Göring N., Durstewitz D.
# Bifurcations and loss jumps in RNN training
# Thirty-seventh Conference on Neural Information Processing Systems, 2023
# https://openreview.net/forum?id=QmPf29EHyI

import numpy as np
from itertools import combinations, chain


def powerset(iterable):
    """get powerset"""
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def construct_relu_matrix(number_quadrant: int, dim: int):
    """
    Get D matrix indicating which relu neurons are above threshold
    by converting number_quadrant to binary
    """
    quadrant_index = format(number_quadrant, f"0{dim}b")[::-1]
    return np.diag(np.array([bool(int(bit)) for bit in quadrant_index]))


def construct_relu_matrix_list(dim: int, order: int):
    """
    Construct a list of D matrices for a random sequence of subregions
    """
    relu_matrix_list = np.empty((dim, dim, order))
    for i in range(order):
        # randomly sample a subregion
        n = int(np.floor(np.random.rand(1)[0] * (2**dim)))
        relu_matrix_list[:, :, i] = construct_relu_matrix(n, dim)
    return relu_matrix_list


def get_cycle_point_candidate(A, W1, W2, h1, h2, D_list, order):
    """
    Get the candidate for a cycle point by solving the cycle equation
    (finds 'virtual' fixed points of order times iterated system)
    """
    z_factor, h1_factor, h2_factor = get_factors(A, W1, W2, D_list, order)
    try:
        inverse_matrix = np.linalg.inv(np.eye(A.shape[0]) - z_factor)
        z_candidate = inverse_matrix.dot(h1_factor.dot(h1) + h2_factor.dot(h2))
        return z_candidate
    except np.linalg.LinAlgError:
        # Not invertible
        return None


def get_factors(A, W1, W2, D_list, order):
    """
    iteratively applying RNN gives us the factors of the cycle equation
    """
    hidden_dim = W2.shape[0]
    latent_dim = W1.shape[0]
    factor_z = np.eye(A.shape[0])
    factor_h1 = np.eye(A.shape[0])
    factor_h2 = W1.dot(D_list[:, :, 0]).dot(np.eye(hidden_dim))
    for i in range(order - 1):
        factor_z = (A + W1.dot(D_list[:, :, i]).dot(W2)).dot(factor_z)
        factor_h1 = (A + W1.dot(D_list[:, :, i + 1]).dot(W2)).dot(factor_h1) + np.eye(
            A.shape[0]
        )
        factor_h2 = (A + W1.dot(D_list[:, :, i + 1]).dot(W2)).dot(factor_h2) + W1.dot(
            D_list[:, :, i + 1]
        )
    factor_z = (A + W1.dot(D_list[:, :, order - 1]).dot(W2)).dot(factor_z)
    return factor_z, factor_h1, factor_h2


def get_latent_time_series(time_steps, A, W1, W2, h1, h2, dz, z_0=None):
    """
    Generate the time series by iteravely applying the RNN
    """
    if z_0 is None:
        z = np.random.randn(dz)
    else:
        z = z_0
    trajectory = [z]

    for t in range(1, time_steps):
        z = latent_step(z, A, W1, W2, h1, h2)
        trajectory.append(z)
    return trajectory


def latent_step(z, A, W1, W2, h1, h2):
    """
    One step of a piecewise linear RNN
    """
    return A.dot(z) + W1.dot(np.maximum(W2.dot(z) + h2, 0)) + h1


def get_eigvals(A, W1, W2, D_list, order):
    """
    Get the eigenvalues for all the points along the trajectory to learn about the stability
    """
    e = np.eye(A.shape[0])
    for i in range(order):
        e = (np.diag(A) + W1.dot(D_list[:, :, i]).dot(W2)).dot(e)
    return np.linalg.eigvals(e)


def scy_fi(
    A,
    W1,
    W2,
    h1,
    h2,
    order,
    found_lower_orders,
    outer_loop_iterations=300,
    inner_loop_iterations=100,
    round_dec=2,
    constrain=False,
    n_inverses_max=1000,
):
    """
    Heuristic algorithm for calculating FP/k-cycle
    Here, with the additional possibility of constraining the search space

    Args:
        A: np.array (d_z x d_z), latent decay matrix
        W1: np.array (d_z x N), right connectivity vectors
        W2: np.array (N x d_z), left connectivity vectors
        h1: np.array (d_z), latent bias
        h2: np.array (N), hidden bias
        order: int, order of the cycle
        outer_loop_iterations: int, number of outer loop iterations
        inner_loop_iterations: int, number of inner loop iterations
        constrain: bool, whether to constrain the search space based on analytical solution
        n_inverses_max: int, maximum number of inverses
    Returns:
        found_lower_orders: list of found lower orders
        found_eigvals: list of found eigenvalues
        n_inverses: int, number of inverses
    """
    hidden_dim = h2.shape[0]
    latent_dim = h1.shape[0]
    cycles_found = []
    eigvals = []
    n_inverses = 0

    # constrain to subregions that can contain fixed points
    if constrain:
        N = W2.shape[0]
        R = W2.shape[1]
        # First solve for all intersection of hyperplanes
        intersect_inds = np.array(list(combinations(np.arange(N), R)))

        n_Ds_initial = len(list(powerset(range(R)))) * len(intersect_inds)
        D_list = np.zeros((n_Ds_initial, N), dtype="uint8")
        it = 0
        for inds in intersect_inds:
            b_hat = -h2[inds]
            U_hat = W2[inds]
            n_inverses += 1
            z = np.linalg.solve(U_hat, b_hat)
            # Find all subspaces bordering to this intersection
            x = W2 @ z + h2
            D_init = np.array(x > 0).astype("uint8")
            D_init[inds] = 0
            D_list[it] = D_init
            it += 1
            D_inds = list(powerset(inds))[1:]
            for D_ind in D_inds:
                D = np.copy(D_init)
                D[np.array(D_ind)] = 1
                D_list[it] = D
                it += 1

    i = -1
    while i < outer_loop_iterations and n_inverses < n_inverses_max:
        i += 1
        if constrain:
            # Sample from subspace that can contain fixed points
            ind = np.random.randint(low=0, high=len(D_list))
            relu_matrix_list = np.expand_dims(np.diag(D_list[ind]), -1)
        else:
            relu_matrix_list = construct_relu_matrix_list(hidden_dim, order)
        difference_relu_matrices = 1
        c = 0
        while c < inner_loop_iterations and n_inverses < n_inverses_max:
            c += 1
            z_candidate = get_cycle_point_candidate(
                A, W1, W2, h1, h2, relu_matrix_list, order
            )
            n_inverses += 1
            if z_candidate is not None:
                trajectory = get_latent_time_series(
                    order, A, W1, W2, h1, h2, latent_dim, z_0=z_candidate
                )
                trajectory_relu_matrix_list = np.empty((hidden_dim, hidden_dim, order))
                for j in range(order):
                    trajectory_relu_matrix_list[:, :, j] = np.diag(
                        (W2.dot(trajectory[j]) + h2) > 0
                    )
                for j in range(order):
                    difference_relu_matrices = np.sum(
                        np.abs(
                            trajectory_relu_matrix_list[:, :, j]
                            - relu_matrix_list[:, :, j]
                        )
                    )
                    if difference_relu_matrices != 0:
                        break
                    if found_lower_orders:
                        if np.round(trajectory[0], decimals=2) in np.round(
                            np.array(found_lower_orders).flatten(), decimals=2
                        ):
                            difference_relu_matrices = 1
                            break
                if difference_relu_matrices == 0:
                    if not np.any(
                        np.isin(
                            np.round(trajectory[0], round_dec),
                            np.round(cycles_found, round_dec),
                        )
                    ):
                        e = get_eigvals(A, W1, W2, relu_matrix_list, order)
                        cycles_found.append(trajectory)
                        eigvals.append(e)
                        i = 0
                        c = 0
                if np.array_equal(relu_matrix_list, trajectory_relu_matrix_list):
                    if constrain:
                        # Sample from subspace that can contain fixed points
                        ind = np.random.randint(low=0, high=len(D_list))
                        relu_matrix_list = np.expand_dims(np.diag(D_list[ind]), -1)
                    else:
                        relu_matrix_list = construct_relu_matrix_list(hidden_dim, order)
                else:
                    relu_matrix_list = trajectory_relu_matrix_list
            else:
                relu_matrix_list = construct_relu_matrix_list(hidden_dim, order)
    return cycles_found, eigvals, n_inverses


def run_scify(
    A,
    W1,
    W2,
    h1,
    h2,
    order=1,
    outer_loop_iterations=300,
    inner_loop_iterations=100,
    round_dec=2,
    constrain=True,
    n_inverses_max=1000,
):
    """
    Run the scify algorithm for a given order
    Args:
        A: np.array (d_z x d_z), latent decay matrix
        W1: np.array (d_z x N), right connectivity vectors
        W2: np.array (N x d_z), left connectivity vectors
        h1: np.array (d_z), latent bias
        h2: np.array (N), hidden bias
        order: int, order of the cycle
        outer_loop_iterations: int, number of outer loop iterations
        inner_loop_iterations: int, number of inner loop iterations
        constrain: bool, whether to constrain the search space based on analytical solution
        n_inverses_max: int, maximum number of inverses
    Returns:
        found_lower_orders: list, list of found lower orders
        found_eigvals: list, list of found eigenvalues
        n_inverse: list, int, number of inverses
    """

    found_lower_orders = []
    found_eigvals = []
    n_inverses = []
    for i in range(1, order + 1):
        cycles_found, eigvals, n_inverses_ = scy_fi(
            A,
            W1,
            W2,
            h1,
            h2,
            i,
            found_lower_orders,
            outer_loop_iterations=outer_loop_iterations,
            inner_loop_iterations=inner_loop_iterations,
            round_dec=round_dec,
            constrain=constrain,
            n_inverses_max=n_inverses_max,
        )
        found_lower_orders.append(cycles_found)
        found_eigvals.append(eigvals)
        n_inverses.append(n_inverses_)
    return found_lower_orders, found_eigvals, n_inverses
