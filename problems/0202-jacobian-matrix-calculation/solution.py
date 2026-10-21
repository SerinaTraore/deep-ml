import numpy as np


def jacobian_matrix(f, x: list[float], h: float = 1e-5) -> list[list[float]]:
    """Compute the Jacobian matrix using numerical differentiation.

    Args:
        f: Function that takes a list and returns a list
        x: Point at which to evaluate the Jacobian
        h: Step size for finite differences

    Returns:
        Jacobian matrix as list of lists
    """
    n = len(x)
    f_x = f(x)
    m = len(f_x)

    jacobian = []
    for _ in range(m):
        jacobian.append([0.0] * n)

    for j in range(n):
        x_plus = list(x)
        x_minus = list(x)

        x_plus[j] += h
        x_minus[j] -= h

        f_plus = f(x_plus)
        f_minus = f(x_minus)

        for i in range(m):
            jacobian[i][j] = (f_plus[i] - f_minus[i]) / (2 * h)

    return jacobian