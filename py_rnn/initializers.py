import numpy as np


def initialize_w_rec(params):
    """
    Initializes (full rank) recurrent weight matrix

    Args:
        params: python dictionary containing network parameters

    Returns:
        w_rec: recurrent weight matrix, numpy array of shape [n_rec, n_rec]
        dale_mask: diagonal matrix indicating exh or inh, shape [n_rec, n_rec]
    """

    w_rec = np.zeros((params["n_rec"], params["n_rec"]), dtype=np.float32)
    dale_mask = np.eye(params["n_rec"], dtype=np.float32)
    rec_idx = np.where(
        np.random.rand(params["n_rec"], params["n_rec"]) < params["p_rec"]
    )

    # initialize with weights drawn from either Gaussian or Gamma distribution
    if params["w_rec_dist"] == "gauss":
        w_rec[rec_idx[0], rec_idx[1]] = (
            np.random.normal(0, 1, len(rec_idx[0]))
            * params["spectr_rad"]
            / np.sqrt(params["p_rec"] * params["n_rec"])
        )
    elif params["w_dist"] == "gamma":
        w_rec[rec_idx[0], rec_idx[1]] = np.random.gamma(2, 0.5, len(rec_idx[0]))
        if params["spectr_norm"] == False:
            print(
                "WARNING: analytic normalisation not implemented for gamma, setting spectral normalisation to TRUE"
            )
            params["spectr_norm"] = True
        if params["apply_dale"] == False:
            print(
                "WARNING: Gamma distribution is all positive, use only with Dale's law, setting Dale's law to TRUE"
            )
            params["apply_dale"] == True

    else:
        print("WARNING: initialization not implemented, use Gauss or Gamma")
        print("continuing with Gauss")
        w_rec[rec_idx[0], rec_idx[1]] = (
            np.random.normal(0, 1, len(rec_idx[0]))
            * params["spectr_rad"]
            / np.sqrt(params["p_rec"] * params["n_rec"])
        )

    # set to desired spectral radius
    if params["spectr_norm"]:
        w_rec = (
            params["spectr_rad"]
            * w_rec
            / np.max(np.abs((np.linalg.eigvals(dale_mask.dot(w_rec)))))
        )
    print("spectral_rad: " + str(np.max(abs(np.linalg.eigvals(dale_mask.dot(w_rec))))))
    return w_rec


def initialize_w_inp(params):
    """
    Initializes input weight matrix

    Args:
        params: python dictionary containing network parameters

    Return:
        w_inp: input weight matrix, numpy array of size [n_rec, n_inp]
    """

    w_task = np.zeros((params["n_rec"], params["n_inp"]), dtype=np.float32)
    idx = np.array(
        np.where(np.random.rand(params["n_rec"], params["n_inp"]) < params["p_inp"])
    )
    w_task[idx[0], idx[1]] = np.random.randn(len(idx[0])) * np.sqrt(
        params["scale_w_inp"] / params["p_inp"]
    )

    return w_task.T


def initialize_loadings(params):
    """
    Initializes weight matrices for low rank networks

    Args:
        params: python dictionary containing network parameters

    Returns:
        loadings: weight matrices for low rank networks
            numpy array of size [rank * 2 + n_inp + n_out, n_rec]
    """

    n_loading = params["rank"] * 2 + params["n_inp"] + params["n_out"]

    if params["cov"] is None:
        # generate random covariance matrix
        # with n and m correlated to avoid vanishing gradients"
        cov = np.eye(n_loading)
        for i in range(params["rank"]):
            cov[params["n_inp"] + i, params["n_inp"] + params["rank"] + i] = 0.8
            cov[params["n_inp"] + params["rank"] + i, params["n_inp"] + i] = 0.8
    else:
        cov = params["cov"]
    # use cholesky decomposition to draw vectors
    chol_cov = np.float32(np.linalg.cholesky(cov))
    loadings = chol_cov @ np.float32(np.random.randn(n_loading, params["n_rec"]))

    return loadings
