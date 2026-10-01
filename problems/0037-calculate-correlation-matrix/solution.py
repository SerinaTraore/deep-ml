import numpy as np


def calculate_correlation_matrix(X, Y=None):
    X = np.asarray(X, dtype=float)

    # Cas 1 : Seul X est fourni -> Auto-corrélation (taille n_features_X x n_features_X)
    if Y is None:
        X_c = X - np.mean(X, axis=0)
        norm_X = np.linalg.norm(X_c, axis=0)
        denom = np.outer(norm_X, norm_X)
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = (X_c.T @ X_c) / denom
        return corr

    # Cas 2 : Y est fourni -> Corrélation croisée entre X et Y (taille n_features_X x n_features_Y)
    Y = np.asarray(Y, dtype=float)
    X_c = X - np.mean(X, axis=0)
    Y_c = Y - np.mean(Y, axis=0)

    norm_X = np.linalg.norm(X_c, axis=0)
    norm_Y = np.linalg.norm(Y_c, axis=0)

    # Dénominateur croisé : norm_X[i] * norm_Y[j]
    denom = np.outer(norm_X, norm_Y)

    with np.errstate(divide="ignore", invalid="ignore"):
        corr = (X_c.T @ Y_c) / denom

    return corr