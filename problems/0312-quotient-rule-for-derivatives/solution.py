import numpy as np

def quotient_rule_derivative(g_coeffs: list, h_coeffs: list, x: float) -> float:
    """
    Compute the derivative of f(x) = g(x)/h(x) at point x using the quotient rule.
    
    Args:
        g_coeffs: Coefficients of numerator polynomial in descending order
        h_coeffs: Coefficients of denominator polynomial in descending order
        x: Point at which to evaluate the derivative
        
    Returns:
        The derivative value f'(x)
    """
    g_val = np.polyval(g_coeffs, x)
    h_val = np.polyval(h_coeffs, x)

    g_prime_coeffs = np.polyder(g_coeffs)
    h_prime_coeffs = np.polyder(h_coeffs)

    g_prime_val = np.polyval(g_prime_coeffs, x)
    h_prime_val = np.polyval(h_prime_coeffs, x)

    numerator = (g_prime_val * h_val) - (h_prime_val * g_val)
    denominator = h_val ** 2
    return  np.rnumerator / denominator
    pass