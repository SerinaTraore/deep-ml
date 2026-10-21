import numpy as np


def softmax_derivative(x: list[float]) -> list[list[float]]:
    """Compute the Jacobian matrix of the softmax function.

    Args:
        x: Input vector of real numbers

    Returns:
        Jacobian matrix J where J[i][j] = d(softmax_i)/d(x_j)
    """
    x_arr = np.array(x, dtype=float)
    # Subtract max for numerical stability
    exp_x = np.exp(x_arr - np.max(x_arr))
    s = exp_x / np.sum(exp_x)

    # J[i, j] = s[i] * (1 - s[i]) if i == j else -s[i] * s[j]
    # Represented concisely using outer product: diag(s) - s * s^T
    jacobian = np.diag(s) - np.outer(s, s)

    return jacobian.tolist()