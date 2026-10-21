import numpy as np


def activation_derivatives(x: float) -> dict[str, float]:
    """Compute the derivatives of Sigmoid, Tanh, and ReLU at a given point x.

    Args:
        x: Input value

    Returns:
        Dictionary with keys 'sigmoid', 'tanh', 'relu' and their derivative values
    """
    # Sigmoid derivative: σ'(x) = σ(x) * (1 - σ(x))
    sigmoid_val = 1.0 / (1.0 + np.exp(-x))
    d_sigmoid = sigmoid_val * (1.0 - sigmoid_val)

    # Tanh derivative: tanh'(x) = 1 - tanh²(x)
    tanh_val = np.tanh(x)
    d_tanh = 1.0 - tanh_val**2

    # ReLU derivative: relu'(x) = 1 if x > 0 else 0 (using subgradient 0 at x = 0)
    d_relu = 1.0 if x > 0 else 0.0

    return {
        "sigmoid": float(d_sigmoid),
        "tanh": float(d_tanh),
        "relu": float(d_relu),
    }