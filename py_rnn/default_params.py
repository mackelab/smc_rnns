def get_default_params(n_rec=100):
    """Initialize default parameters for the model and training"""

    model_params = {
        "nonlinearity": "relu",  # activation function
        "rank": 2,  # rank, set to 0 for full rank
        "n_inp": 1,  # amount of input units
        "p_inp": 1,  # probability of connection input
        "n_rec": n_rec,  # amount of recurrent units
        "p_rec": 1,  # probability of connection recurrent
        "n_out": 1,  # amount of output units
        "scale_w_inp": 1,  # scale input weights
        "scale_w_out": 1,  # scale output weights
        "w_rec_dist": "gauss",  # recurrent weight dist, gauss or gamma
        "spectr_rad": 1,  # gain param, recurrent weights
        "spectr_norm": True,  # use spectral normalisation on rec weights
        "train_w_inp": True,  # train input weights
        "train_w_inp_scale": False,  # train input scaling factor
        "train_w_rec": True,  # train recurrent weights
        "train_b_rec": True,  # train recurrent bias
        "train_taus": False,  # train time constants
        "train_m": False,  # train m
        "train_n": True,
        "scale_n": 1,
        "scale_m": 1,
        "loadings": None,
        "cov": None,
        "train_w_out": True,  # train output weights
        "train_w_out_scale": False,  # train output scaling factor
        "train_x0": True,  # train initial state
        "tau_lims": [100],  # tau limits (min, max) or (value) in ms
        "project_taus": "sigmoid",  # choice of projection map to keep within limits ("exp", "sigmoid" or "clip")
        "tau_mean": 100,  # if tau distribution, specify mean
        "tau_std": 1,  # if tau distribution, specify std
        "dt": 10,  # timestep in ms
        "noise_std": 0.05,  # noise std
        "scale_x0": 0.1,  # std of initial state, if gaussian
        "conn_mask": None,  # connection mask (tensor of size n_rec*n_rec)
        "dale_mask": None,  # dale mask (tensor of size n_rec*n_rec with only -1 and 1's on diagonal)
    }

    training_params = {
        "n_epochs": 500,  # number of passes through possible trials
        "lr": 10e-3,  # learning rate
        "batch_size": 4,  # batch size
        "clip_gradient": 1,  # to avoid explosion of gradients
        "cuda": False,  # train on GPU if True
        "loss_fn": "mse",  # loss function (mse, cos or none)
        "optimizer": "adam",  # optimizer (adam)
        "osc_reg_cost": 0,  # oscillatory regularisation weight
        "osc_reg_freq": 2,  # oscillatory regularisation frequency
        "offset_reg_cost": 0,  # offset regularisation cost
        "l2_rates_reg": 0,  # l2 regularisation on rates
    }
    return model_params, training_params
