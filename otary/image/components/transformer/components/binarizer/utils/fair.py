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


def responsibility(x, mu_t, mu_b, var_t, var_b, omega):
    p_t = omega * gaussian_pdf(x, mu=mu_t, var=var_t)
    p_b = (1 - omega) * gaussian_pdf(x, mu=mu_b, var=var_b)
    return p_t / (p_t + p_b + 1e-9)


def expectation_maximization(
    x: NDArray,
    rel_tol: float = 1e-2,
    max_iter: int = 100,
):
    """EM algorithm using both original image and edges image.
    This is vectorized implementation that can process multiple patches at once.
    It assumes a x input of shape:
    - (N, n, n) or 3D input: N patches of size (n, n)
    - (N, m) or 2D input: N vectors of size m

    Args:
        x (NDArray): input images
        rel_tol (float, optional): _description_. Defaults to 1e-3.
        max_iter (int, optional): _description_. Defaults to 100.
    """
    # ensure x is always 3D: (n, m, c) to make the code compatible with 2D tensor input
    squeeze_back = False
    if x.ndim == 2:
        x = x[..., np.newaxis]
        squeeze_back = True

    n_edges = x.shape[0]

    # EM initialization - mu (mean), sigma (std), omega (mixing coefficient)
    mu_t: NDArray = np.min(x, axis=(1, 2))[:, np.newaxis, np.newaxis]
    mu_b: NDArray = np.mean(x, axis=(1, 2))[:, np.newaxis, np.newaxis] + mu_t / 2
    omega = (np.random.random(size=(n_edges, 1, 1)) + 4) / 5
    gamma = np.where(x < (mu_t + mu_b) / 2, 0.75, 0.25)
    var_t, var_b = 1, 1

    for _ in range(max_iter):
        # E-step
        gamma = responsibility(x, mu_t, mu_b, var_t, var_b, omega)

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

        # check convergence
        if (
            np.all(np.abs(omega - _omega) < rel_tol).astype(bool)
            and np.all(np.abs(mu_t - _mu_t) < rel_tol).astype(bool)
            and np.all(np.abs(mu_b - _mu_b) < rel_tol).astype(bool)
            and np.all(np.abs(var_t - _var_t) < rel_tol).astype(bool)
            and np.all(np.abs(var_b - _var_b) < rel_tol).astype(bool)
        ):
            break

        # update parameters
        mu_t, mu_b, var_t, var_b, omega = _mu_t, _mu_b, _var_t, _var_b, _omega

    # swap params so that _t always refers to text (darker pixels) and _b to background
    swap_mask = mu_t > mu_b
    mu_t_old, mu_b_old = mu_t, mu_b
    mu_t = np.where(swap_mask, mu_b_old, mu_t_old)
    mu_b = np.where(swap_mask, mu_t_old, mu_b_old)
    omega = np.where(swap_mask, 1 - omega, omega)
    gamma = np.where(swap_mask, 1 - gamma, gamma)

    if np.mean(mu_t) >= np.mean(mu_b):
        # swap variances too based on mean values
        var_t, var_b = var_b, var_t

    # restore shape if input was 2D
    if squeeze_back:
        gamma = gamma.squeeze(-1)

    return gamma


def threshold_sfair(
    img: NDArray,
    k: float = 1.0,
    alpha: float = 0.5,
    n: int = 15,
    max_iter: int = 10,
    em_thining_factor: float = 1.0,
    unknown_label: float = 0.5,
):
    """S-FAIR thresholding method.
    This typically serves as a building block for the FAIR method.
    However it can be also used as a standalone binarization method.

    Args:
        img (NDArray): input image
        k (float, optional): _description_. Defaults to 1.0.
        alpha (float, optional): It defines the ratio to compute the lower threshold
            in the 1st step of the S-FAIR step. It is generally in [0.3, 0.5].
            Defaults to 0.4.
        n (int, optional): window size for the EM algorithm and hence to cluster
            background and foreground pixels around edge pixels.
            This parameter is important as a higher value will make the method
            more robust to noise but also more computationally expensive and slow.
            Defaults to 51.
        em_max_iter (int, optional): maximum number of iterations for the EM algorithm.
    """
    # Step 1 of S-FAIR - Text area detection
    gm = gradient_magnitude(img=img, window_size=3)
    T_o = cv2.threshold(gm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    T_u = k * T_o  # T_u stands for upper threshold
    T_l = alpha * T_o  # T_l stands for lower threshold
    img_edged = cv2.Canny(image=img, threshold1=T_l, threshold2=T_u)  # values 0 or 255
    s = np.column_stack(np.where(img_edged == 255))  # edge pixel coordinates

    # to speed up EM, we can downsample the number of edge pixels to process
    # thining must be in [n // 2, 1]
    thining = n // int(n * (1 - em_thining_factor) + 2 * em_thining_factor)
    s = s[::thining]  # downsample edge pixels to speed up EM

    # Step 2 of S-FAIR - Model estimation around edges
    # edges can be easily identified as they are 255 pixels in im_edges
    # get patches for vectorized computation - trick do padding to get all patches
    # of same size even when the center of edge pixel is at border
    pad = n // 2
    img_pad = cv2.copyMakeBorder(img, pad, pad, pad, pad, borderType=cv2.BORDER_DEFAULT)
    patches = np.lib.stride_tricks.sliding_window_view(x=img_pad, window_shape=(n, n))
    x = patches[s[:, 0], s[:, 1]]  # shape (n_edges, n, n)
    gamma = expectation_maximization(x=x, max_iter=max_iter)

    # compute the mean responsability for each pixel
    # to get a smoother result (as each pixel can be in multiple patches)
    resp_sum = np.zeros_like(img_pad, dtype=np.float32)
    resp_count = np.zeros_like(img_pad, dtype=np.float32)
    for i, (r, c) in enumerate(s):
        resp_sum[r : r + n, c : c + n] += gamma[i]
        resp_count[r : r + n, c : c + n] += 1
    resp_count[resp_count == 0] = 1  # avoid division by zero
    gamma_tilde = resp_sum / resp_count  # average the responsibilities

    # compute z_i = 1 if gamma_tilde > 0.5 else 0
    z = np.where(gamma_tilde > 0.5, 1, 0)
    z[resp_sum == 0] = unknown_label
    z = z[pad:-pad, pad:-pad]  # remove padding to get back to original img size

    return z


def dilate_binary_cross(mask: np.ndarray, distance: int = 1) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8)

    # cross-shaped kernel (4-connectivity) - called diamond or D_5 in FAIR paper
    _kernel = (2 * distance + 1, 2 * distance + 1)
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_CROSS, ksize=_kernel)

    dilated = cv2.dilate(src=mask, kernel=kernel, iterations=1)
    return dilated


def threshold_fair(
    img: NDArray, n: int = 15, max_iter: int = 100, em_thining_factor: float = 1.0
):
    """FAIR thresholding method.

    Args:
        img (NDArray): input image
        n (int, optional): window size for the EM algorithm and hence to cluster
            background and foreground pixels around edge pixels.
            This parameter is important as a higher value will make the method
            more robust to noise but also more computationally expensive and slow.
            Defaults to 51.
        em_max_iter (int, optional): maximum number of iterations for the EM algorithm.
    """
    UNKNOWN_LABEL = 0.5

    # Step 1 - Double Thresholding: compute S-FAIR twice with different k
    z1 = threshold_sfair(
        img=img,
        k=1.44,
        alpha=0.5,
        n=n,
        max_iter=max_iter,
        em_thining_factor=em_thining_factor,
        unknown_label=UNKNOWN_LABEL,
    )
    z2 = threshold_sfair(
        img=img,
        k=1.6,
        alpha=0.5,
        n=n,
        max_iter=max_iter,
        em_thining_factor=em_thining_factor,
        unknown_label=UNKNOWN_LABEL,
    )

    # Step 2 - Merging: combine the two S-FAIR results
    I_m = np.max([z1, z2], axis=0)  # values in {0, 0.5, 1}

    # Step 3 - Post-Filtering process
    # Step 3.a. - remove stains connected components only surrounded by unknown pixels
    # TODO

    # Step 3.b. - correct misclassified text pixels
    z_ti = np.where(I_m == 0, 1, 0)
    z_ui = np.where(I_m == UNKNOWN_LABEL, 1, 0)
    z_pti = np.where((z_ti) & (dilate_binary_cross(z_ui, distance=2)), 1, 0)
    z_uti = np.where((z_ui) & (dilate_binary_cross(z_ti, distance=2)), 1, 0)
    if np.sum(z_pti) > np.sum(z_uti):
        pass

    # N_f(S) = N(S) INTERSECTION (Z_pti UNION Z_uti)
    # since EM is robust to identical values we can set element not in N(s) to some
    # pre-existing value in the window

    # Step 4. - Final labeling

    return I_m
