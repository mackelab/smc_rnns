from itertools import combinations, chain
import numpy as np


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def find_fixed_points_analytic(a, V, U, hz, h, d=1):
    """
    Find fixed points of the model
    Args:
        a: numpy array of shape (R,) decay
        V: numpy array of shape (R,N) scaled left singular vectors
        U: numpy array of shape (N,R) right singular vectors
        hz: numpy array of shape (R,) latent bias (assumed to be substracted!)
        h: numpy array of shape (N,) neuron bias
    Returns:
        D_list: numpy array of shape (n_Ds,N) containing all subspaces
        D_inds: list of indices of subspaces in D_list that are fixed points
        z_list: list of fixed points
        n_inverses: number of inverses
    """

    n_inverses = 0
    N = U.shape[0]
    R = U.shape[1]

    # First solve for all intersection of hyperplanes
    intersect_inds = np.array(list(combinations(np.arange(N), R)))
    print(len(intersect_inds))

    par_inds = []
    if d == 2:
        ni = N // 2
        for i, el in enumerate(intersect_inds):
            if el[0] == el[1] + ni or el[1] == el[0] + ni:
                par_inds.append(i)
        intersect_inds = np.delete(intersect_inds, par_inds, axis=0)
        print("removed parallel lines")
        print(len(intersect_inds))

    n_Ds_initial = len(list(powerset(range(R)))) * len(intersect_inds)
    print(len(list(powerset(range(R)))))
    D_list = np.zeros((n_Ds_initial, N), dtype="uint8")
    it = 0
    for inds in intersect_inds:
        b_hat = h[inds]
        U_hat = U[inds]
        n_inverses += 1
        z = np.linalg.solve(U_hat, b_hat)
        # Find all subspaces bordering to this intersection
        x = U @ z - h
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
    # Throw away duplicate subspaces
    print(D_list.shape)
    D_list = np.unique(D_list, axis=0)
    print(D_list.shape)

    # Finally solve for fixed points
    z_list = []
    D_inds = []
    for D_ind, D_init in enumerate(D_list):

        A = -np.eye(R) + np.diag(a) + V @ (D_init[:, None] * U)
        b = V @ (D_init * h) + hz
        z_hat = np.linalg.solve(A, b)
        n_inverses += 1

        x_hat = U @ z_hat - h
        if np.allclose(D_init, np.array(x_hat > 0).astype("uint8")):
            print("Found a fixed point")
            print(z_hat)
            z_list.append(z_hat)
            D_inds.append(D_ind)
    print("Done, found " + str(len(z_list)) + " fixed points")
    return D_list, D_inds, z_list, n_inverses
