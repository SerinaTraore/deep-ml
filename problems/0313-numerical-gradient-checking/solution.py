import numpy as np


def numerical_gradient_check(f, x, analytical_grad, epsilon=1e-7):
    """Perform numerical gradient checking using centered finite differences.

    Args:
        f: A function that takes a numpy array and returns a scalar
        x: numpy array, the point at which to check gradient
        analytical_grad: numpy array, the analytically computed gradient
        epsilon: float, small value for finite difference approximation

    Returns:
        tuple: (numerical_grad, relative_error)
    """
    x = np.asarray(x, dtype=np.float64)
    analytical_grad = np.asarray(analytical_grad, dtype=np.float64)

    num_grad = np.zeros_like(x)

    # Calcul du gradient numérique composante par composante
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        old_val = x[idx]

        # f(x + epsilon)
        x[idx] = old_val + epsilon
        fx_plus = f(x)

        # f(x - epsilon)
        x[idx] = old_val - epsilon
        fx_minus = f(x)

        # Restauration de la valeur initiale de x
        x[idx] = old_val

        # Différence finie centrée
        num_grad[idx] = (fx_plus - fx_minus) / (2 * epsilon)

        it.iternext()

    # Calcul de l'erreur relative normalisée
    numerator = np.linalg.norm(num_grad - analytical_grad)
    denominator = np.linalg.norm(num_grad) + np.linalg.norm(analytical_grad)

    # Évite la division par zéro si les deux gradients sont nuls
    if denominator == 0:
        relative_error = 0.0
    else:
        relative_error = numerator / denominator

    return num_grad, relative_error