import numpy as np


def Relu_derivative(x):
    return np.array(x > 0).astype("float")


def PL_Jacobian(V, U, h, a, z):
    """
    Compute the jacobian of the piecewise linear model
    Args:
        V: numpy array of shape (R,N)
        U: numpy array of shape (N,R)
        h: numpy array of shape (N,)
    Returns:
        J: numpy array of shape (N,N) representing the jacobian
    """
    x = U @ z
    D1 = np.diag(Relu_derivative(x + h))
    J = a @ np.eye(len(z)) + V @ (D1) @ U
    return J
