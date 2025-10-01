"""
Expectation-Maximation algorithm for a mixture of two gaussians
"""

import numpy as np
from numpy.typing import NDArray


def gaussian_pdf(
    x: NDArray | float,
    mu: NDArray | float = 0.0,
    var: NDArray | float = 1.0,
) -> NDArray | float:
    """Comute the PDF of a gaussian.

    Args:
        x (NDArray): array
        mu (NDArray | float, optional): mean. Defaults to 0.0.
        var (NADArray | float, optional): variance. Defaults to 1.0.

    Returns:
        NDArray | float: output PDF for value x
    """
    return (1.0 / (var * np.sqrt(2 * np.pi) + 1e-9)) * np.exp(
        -0.5 * (x - mu) ** 2 / (var + 1e-9)
    )


def responsibility(
    x: NDArray,
    mu_t: NDArray | float,
    mu_b: NDArray | float,
    var_t: NDArray | float,
    var_b: NDArray | float,
    omega: NDArray,
) -> NDArray:
    """Compute the responsibility considering a mixture of two gaussians
    which is the hypothesis in the FAIR algorithm

    Args:
        x (NDArray | float): input value
        mu_t (NDArray | float): mean value of the first gaussian
        mu_b (NDArray | float): mean value of the second gaussian
        var_t (NDArray | float): variance value of the first gaussian
        var_b (NDArray | float): variance value of the second gaussian
        omega (NDArray | float): mixing coefficient

    Returns:
        NDArray: responsibility values
    """
    p_t = omega * gaussian_pdf(x, mu=mu_t, var=var_t)
    p_b = (1 - omega) * gaussian_pdf(x, mu=mu_b, var=var_b)
    return p_t / (p_t + p_b + 1e-9)


def expectation_maximization(
    x: NDArray,
    tol: float = 1e-2,
    max_iter: int = 100,
) -> NDArray:
    """EM algorithm for a mixture of two gaussians.

    A detail implementation can be found at page 5 in the following paper:
    https://arxiv.org/pdf/1901.06708

    This is vectorized implementation that can process multiple patches at once.
    It assumes a x input of shape:
    - (N, w, n) or 3D input: N patches of size (w, n). Generally the window is a square
        so it would be (N, n, n).

    Args:
        x (NDArray): input image or patches
        tol (float, optional): tolerance to check convergence.
            A higher value will make the algorithm more robust to noise but
            also more computationally expensive.
            Defaults to 1e-3.
        max_iter (int, optional): Maximum number of iterations. Typically 10-100
            is enough. There is no need to reach a full convergence to have a
            good result.
            Defaults to 100.
    """
    n_edges = x.shape[0]

    # EM initialization - mu (mean), sigma (std), omega (mixing coefficient)
    mu_t: NDArray = np.min(x, axis=(1, 2))[:, np.newaxis, np.newaxis]
    mu_b: NDArray = np.mean(x, axis=(1, 2))[:, np.newaxis, np.newaxis] + mu_t / 2
    omega = (np.random.random(size=(n_edges, 1, 1)) + 4) / 5
    var_t, var_b = 1, 1

    for _ in range(max_iter):
        # E-step
        gamma: NDArray = responsibility(
            x=x, mu_t=mu_t, mu_b=mu_b, var_t=var_t, var_b=var_b, omega=omega
        )

        # M-step or classic MLE
        # parameters with underscore are the updated ones
        ngamma = 1 - gamma
        _omega = np.mean(gamma, axis=(1, 2), keepdims=True)
        _mu_t = np.sum(gamma * x, axis=(1, 2), keepdims=True) / (
            np.sum(gamma, axis=(1, 2), keepdims=True) + 1e-9
        )
        _mu_b = np.sum(ngamma * x, axis=(1, 2), keepdims=True) / (
            np.sum(ngamma, axis=(1, 2), keepdims=True) + 1e-9
        )
        _var_t = np.sum(gamma * (x - mu_t) ** 2) / (np.sum(gamma) + 1e-9)
        _var_b = np.sum(ngamma * (x - mu_b) ** 2) / (np.sum(ngamma) + 1e-9)

        converged = (
            np.all(np.abs(omega - _omega) < tol).astype(bool)
            and np.all(np.abs(mu_t - _mu_t) < tol).astype(bool)
            and np.all(np.abs(mu_b - _mu_b) < tol).astype(bool)
            and np.all(np.abs(var_t - _var_t) < tol).astype(bool)
            and np.all(np.abs(var_b - _var_b) < tol).astype(bool)
        )

        if converged:
            break

        # update parameters
        mu_t, mu_b, var_t, var_b, omega = _mu_t, _mu_b, _var_t, _var_b, _omega

    # swap params so that _t always refers to text (darker pixels) and _b to background
    if np.mean(mu_t) >= np.mean(mu_b):
        # swap variances too based on mean values
        var_t, var_b = var_b, var_t
    swap_mask = mu_t > mu_b
    mu_t_old, mu_b_old = mu_t, mu_b
    mu_t = np.where(swap_mask, mu_b_old, mu_t_old)
    mu_b = np.where(swap_mask, mu_t_old, mu_b_old)
    omega = np.where(swap_mask, 1 - omega, omega)
    gamma = np.where(swap_mask, 1 - gamma, gamma)

    return gamma
