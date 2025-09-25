"""
FAIR thresholding method.
"""

import cv2
import numpy as np
from numpy.typing import NDArray

from otary.image.utils.local import gradient_magnitude


def gaussian_pdf(x, mu=0.0, var=1.0):
    return (1.0 / (var * np.sqrt(2 * np.pi) + 1e-9)) * np.exp(
        -0.5 * (x - mu) ** 2 / (var + 1e-9)
    )


def responsability(x, mu_t, mu_b, var_t, var_b, omega):
    p_t = omega * gaussian_pdf(x, mu=mu_t, var=var_t)
    p_b = (1 - omega) * gaussian_pdf(x, mu=mu_b, var=var_b)
    return p_t / (p_t + p_b + 1e-9)


def expectation_maximization(
    img: NDArray,
    s: NDArray,
    window_size: int = 101,
    rel_tol: float = 1e-7,
    max_iter: int = 100,
):
    """EM algorithm using both original image and edges image.

    Args:
        x (NDArray): input image
        s (NDArray): edges pixels coordinates
        rel_tol (float, optional): _description_. Defaults to 1e-3.
        max_iter (int, optional): _description_. Defaults to 100.
    """
    n_edges = s.shape[0]

    # get patches for vectorized computation - trick do padding to get all patches
    # of same size even when the center of edge pixel is at border
    pad = window_size // 2
    x_pad = cv2.copyMakeBorder(img, pad, pad, pad, pad, borderType=cv2.BORDER_DEFAULT)
    patches = np.lib.stride_tricks.sliding_window_view(
        x=x_pad, window_shape=(window_size, window_size)
    )
    x = patches[s[:, 0], s[:, 1]]  # shape (n_edges, window_size, window_size)

    # EM initialization - mu (mean), sigma (std), omega (mixing coefficient)
    mu_t = np.min(x, axis=(1, 2))
    mu_b = np.max(x, axis=(1, 2))
    omega = np.random.random(size=n_edges)
    gamma = np.random.random(size=(n_edges, window_size, window_size))
    var_t, var_b = 1, 1

    # perform the Expectation-Maximization loop
    is_converged = False
    i = 0
    while not is_converged and i < max_iter:
        # E-step
        gamma = responsability(
            x=x,
            mu_t=mu_t[:, np.newaxis, np.newaxis],  # to allow broadcasting
            mu_b=mu_b[:, np.newaxis, np.newaxis],
            var_t=var_t,
            var_b=var_b,
            omega=omega[:, np.newaxis, np.newaxis],
        )  # shape (n_edges, window_size, window_size)

        print(gamma)

        # M-step or classic MLE
        # parameters with underscore are the updated ones
        ngamma = 1 - gamma
        _omega = np.mean(gamma, axis=(1, 2))  # omega = avg of responsabilities = GAP
        _mu_t = np.sum(gamma * x, axis=(1, 2)) / (np.sum(gamma, axis=(1, 2)) + 1e-9)
        _mu_b = np.sum(ngamma * x, axis=(1, 2)) / (np.sum(ngamma, axis=(1, 2)) + 1e-9)
        _var_t = np.sum(gamma * (x - mu_t[:, np.newaxis, np.newaxis]) ** 2) / (
            np.sum(gamma) + 1e-9
        )
        _var_b = np.sum(ngamma * (x - mu_b[:, np.newaxis, np.newaxis]) ** 2) / (
            np.sum(ngamma) + 1e-9
        )

        # avoid exploding values
        # _var_t = np.clip(_var_t, a_min=1e-9, a_max=50)
        # _var_b = np.clip(_var_b, a_min=1e-9, a_max=50)

        # check convergence
        is_converged = (
            np.all(np.abs(omega - _omega) < rel_tol).astype(bool)
            and np.all(np.abs(mu_t - _mu_t) < rel_tol).astype(bool)
            and np.all(np.abs(mu_b - _mu_b) < rel_tol).astype(bool)
            and np.all(np.abs(var_t - _var_t) < rel_tol).astype(bool)
            and np.all(np.abs(var_b - _var_b) < rel_tol).astype(bool)
        )

        # update parameters
        mu_t, mu_b, var_t, var_b, omega = _mu_t, _mu_b, _var_t, _var_b, _omega

        # avoid infinite loop
        i += 1
        print(i)

    return {
        "x": x,
        "mu_t": mu_t,
        "mu_b": mu_b,
        "sigma_t": var_t,
        "sigma_b": var_b,
        "omega": omega,
        "gamma": gamma,
    }


def threshold_fair(img: NDArray, k: float = 1.0, alpha: float = 0.5, n: int = 3):
    """FAIR thresholding method.

    Args:
        img (NDArray): input image
        k (float, optional): _description_. Defaults to 1.0.
        alpha (float, optional): It defines the ratio to compute the lower threshold
            in the 1st step of the S-FAIR step. It is generally in [0.3, 0.5].
            Defaults to 0.4.
    """
    # Step 1 of S-FAIR - Text area detection
    gm = gradient_magnitude(img=img, window_size=3)
    T_o = cv2.threshold(gm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    T_u = k * T_o  # T_u stands for upper threshold
    T_l = alpha * T_o  # T_l stands for lower threshold
    img_edged = cv2.Canny(image=img, threshold1=T_l, threshold2=T_u)  # values 0 or 255
    s = np.column_stack(np.where(img_edged == 255))  # edge pixel coordinates

    # Step 2 of S-FAIR - Model estimation around edges
    # edges can be easily identified as they are 255 pixels in im_edges
    s = expectation_maximization(img=img, s=s[:1])
    return s
