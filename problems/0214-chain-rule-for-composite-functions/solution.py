import numpy as np


def compute_chain_rule_gradient(functions: list[str], x: float) -> float:
    """Compute derivative of composite functions using chain rule.

    Args:
        functions: List of function names (applied right to left)
                  Available: 'square', 'sin', 'exp', 'log'
        x: Point at which to evaluate derivative

    Returns:
        Derivative value at x

    Example:
        ['sin', 'square'] represents sin(x²)
        ['exp', 'sin', 'square'] represents exp(sin(x²))
    """
    # Define primary functions and their derivatives
    funcs = {
        "square": lambda t: t**2,
        "sin": np.sin,
        "exp": np.exp,
        "log": np.log,
    }

    derivs = {
        "square": lambda t: 2 * t,
        "sin": np.cos,
        "exp": np.exp,
        "log": lambda t: 1 / t,
    }

    # 1. Forward pass: evaluate composite values from right to left
    # values[i] stores the result right before applying functions[i]
    values = [x]
    for fn_name in reversed(functions):
        values.append(funcs[fn_name](values[-1]))

    # 2. Backward pass: apply chain rule (multiply derivatives at each stage)
    # The intermediate input for functions[i] is stored at index (len(functions) - 1 - i)
    grad = 1.0
    for i, fn_name in enumerate(reversed(functions)):
        input_val = values[i]
        grad *= derivs[fn_name](input_val)

    return float(grad)