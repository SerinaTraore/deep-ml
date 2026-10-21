import numpy as np
import sympy as sp


def compute_partial_derivatives(
    func_name: str, point: tuple[float, ...]
) -> tuple[float, ...]:
    """Compute partial derivatives of multivariable functions.

    Args:
        func_name: Function identifier
            'poly2d': f(x,y) = x²y + xy²
            'exp_sum': f(x,y) = e^(x+y)
            'product_sin': f(x,y) = x·sin(y)
            'poly3d': f(x,y,z) = x²y + yz²
            'squared_error': f(x,y) = (x-y)²
        point: Point (x, y) or (x, y, z) at which to evaluate

    Returns:
        Tuple of partial derivatives (∂f/∂x, ∂f/∂y, ...) at point
    """
    x, y, z = sp.symbols("x y z")

    match func_name:
        case "poly2d":
            expr = x**2 * y + x * y**2
            vars_ = (x, y)
        case "exp_sum":
            expr = sp.exp(x + y)
            vars_ = (x, y)
        case "product_sin":
            expr = x * sp.sin(y)
            vars_ = (x, y)
        case "poly3d":
            expr = x**2 * y + y * z**2
            vars_ = (x, y, z)
        case "squared_error":
            expr = (x - y) ** 2
            vars_ = (x, y)
        case _:
            raise ValueError(f"Unknown function identifier: {func_name}")

    subs_dict = dict(zip(vars_, point))
    partials = [
        float(sp.diff(expr, var).evalf(subs=subs_dict)) for var in vars_
    ]

    return tuple(partials)