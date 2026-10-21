import numpy as np


def bayes_theorem(priors: list[float], likelihoods: list[float]) -> list[float]:
    """Calculate posterior probabilities using Bayes' Theorem.

    Args:
        priors: Prior probabilities P(H_i) for each hypothesis
        likelihoods: Likelihoods P(E|H_i) for each hypothesis

    Returns:
        Posterior probabilities P(H_i|E) for each hypothesis
    """
    priors_arr = np.array(priors, dtype=float)
    likelihoods_arr = np.array(likelihoods, dtype=float)

    # Probabilité conjointe P(H_i, E) = P(E|H_i) * P(H_i)
    unnormalized_posteriors = likelihoods_arr * priors_arr

    # Évidence / Probabilité totale de l'évènement P(E) = sum(P(E|H_i) * P(H_i))
    p_e = np.sum(unnormalized_posteriors)

    if p_e == 0:
        raise ValueError(
            "L'évidence P(E) est égale à zéro, impossible de normaliser."
        )

    # P(H_i|E) = P(H_i, E) / P(E)
    posteriors = unnormalized_posteriors / p_e

    return posteriors.tolist()