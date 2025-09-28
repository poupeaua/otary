"""
FAIR thresholding method.

Here is the citation for the original paper:
Thibault Lelore, Frédéric Bouchara. FAIR: A Fast Algorithm for document Image
Restoration.
IEEE Transactions on Pattern Analysis and Machine Intelligence, 2013, 35 (8),
pp.2039-2048.
ff10.1109/TPAMI.2013.63ff. ffhal-01479805f
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
    z = np.where(gamma_tilde > 0.5, 1, 0).astype(np.float32)
    z[resp_sum == 0] = unknown_label
    z = z[pad:-pad, pad:-pad]  # remove padding to get back to original img size

    return z


def dilate_binary_cross(mask: NDArray, distance: int = 1) -> NDArray:
    mask = (mask > 0).astype(np.uint8)

    # cross-shaped kernel (4-connectivity) - called diamond or D_5 in FAIR paper
    _kernel = (2 * distance + 1, 2 * distance + 1)
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_CROSS, ksize=_kernel)

    dilated = cv2.dilate(src=mask, kernel=kernel, iterations=1)
    return dilated


def remove_stains(arr: NDArray, stain_max_pixels: int = 50) -> NDArray:
    mask = (arr != 0.5).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(mask, connectivity=8)

    for i in range(1, num_labels):
        component_mask = (labels == i).astype(np.uint8)

        if np.sum(component_mask) >= stain_max_pixels:
            continue  # skip large components

        dilated = cv2.dilate(
            src=component_mask, kernel=np.ones((3, 3), dtype=np.uint8), iterations=1
        )
        border = (dilated - component_mask).astype(bool)

        # If all border pixels are 0.5 remove the stain
        if np.all(arr[border] == 0.5):
            arr[component_mask.astype(bool)] = 0.5

    return arr


def correct_misclassified_text_pixels(
    img: NDArray,
    I_m: NDArray,
    n: int,
    sfair_max_iter: int,
    fair_max_iter: int,
    unknown_label: float = 0.5,
):
    pad = n // 2
    z_pti_prev = np.zeros_like(I_m)
    for i in range(fair_max_iter):
        z_ti = np.where(I_m == 1, 1, 0)
        z_ui = np.where(I_m == unknown_label, 1, 0)
        z_pti = np.where((z_ti) & (dilate_binary_cross(z_ui, distance=2)), 1, 0)

        if i > 0 and np.array_equal(z_pti, z_pti_prev):
            # no change in z_pti => convergence
            break

        z_pui = np.where((z_ui) & (dilate_binary_cross(z_ti, distance=2)), 1, 0)

        # N_f(S) = N(S) INTERSECTION (Z_pti UNION Z_uti)
        z_fs = img * (z_pti | z_pui)

        s = np.column_stack(np.where(z_pti == 1))

        img_pad = cv2.copyMakeBorder(
            z_fs, pad, pad, pad, pad, borderType=cv2.BORDER_DEFAULT
        )
        patches = np.lib.stride_tricks.sliding_window_view(
            x=img_pad, window_shape=(n, n)
        )
        x = patches[s[:, 0], s[:, 1]]  # shape (N, n, n)

        # since EM is robust to identical values we can set element not in N(s) to some
        # pre-existing value in the window
        # we chose the max value which should be a random background pixel value
        max_per_patch = x.max(axis=(1, 2), keepdims=True)
        x = np.where(x != 0, x, max_per_patch)
        gamma = expectation_maximization(x=x, max_iter=sfair_max_iter)
        centers = gamma[:, pad, pad]
        z = np.where(centers > 0.5, 1, 0)
        I_m[s[:, 0], s[:, 1]] = z

        z_pti_prev = z_pti.copy()

    return I_m


def final_labeling(
    I_m: NDArray, unknown_label: float = 0.5, beta: float = 1.0
) -> NDArray:

    mask = (I_m == unknown_label).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(mask, connectivity=8)

    for i in range(1, num_labels):
        component_mask = (labels == i).astype(np.uint8)
        dilated = cv2.dilate(
            src=component_mask, kernel=np.ones((3, 3), dtype=np.uint8), iterations=1
        )
        border = (dilated - component_mask).astype(bool)

        n_text_pixels_border = np.sum(I_m[border] == 1)
        n_background_pixels_border = np.sum(I_m[border] == 0)

        value = 1 if n_text_pixels_border > beta * n_background_pixels_border else 0
        I_m[component_mask.astype(bool)] = value

    return I_m


def threshold_fair(
    img: NDArray,
    sfair_window_size: int = 5,
    sfair_em_max_iter: int = 50,
    sfair_em_thining_factor: float = 1.0,
    sfair_alpha: float = 0.38,
    stain_max_pixels: int = 50,
    postprocess_enabled: bool = True,
    postprocess_max_iter: int = 15,
    postprocess_em_max_iter: int = 10,
    postprocess_window_size: int = 75,
    beta: float = 1.0,
):
    """FAIR thresholding method.

    Two main configurations are interesting with this method:

    >>> # For whole page document
    >>> bin = threshold_fair(
    >>>     img=im.asarray,
    >>>     sfair_window_size=5,
    >>>     sfair_em_max_iter=100,
    >>>     sfair_em_thining_factor=1.0,
    >>>     postprocess_em_max_iter=15,
    >>>     postprocess_max_iter=10,
    >>>     postprocess_window_size=75,
    >>> )

    >>> # High S-FAIR window size -> can detect very thin text components
    >>> # is good for images with text already zoomed in (only a sentence for example)
    >>> bin = threshold_fair(
    >>>     img=im.asarray,
    >>>     sfair_window_size=75,
    >>>     sfair_em_max_iter=15,
    >>>     sfair_em_thining_factor=1.0,
    >>>     postprocess_em_max_iter=15,
    >>>     postprocess_max_iter=10,
    >>>     postprocess_window_size=35,
    >>> )

    >>> # Low S-FAIR window size -> insist on more local text contours
    >>> bin = threshold_fair(
    >>>     img=im.asarray,
    >>>     sfair_window_size=15,
    >>>     sfair_em_max_iter=25,
    >>>     sfair_em_thining_factor=1.0,
    >>>     postprocess_em_max_iter=15,
    >>>     postprocess_max_iter=10,
    >>>     postprocess_window_size=35,
    >>> )

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
        k=1.4,
        alpha=sfair_alpha,
        n=sfair_window_size,
        max_iter=sfair_em_max_iter,
        em_thining_factor=sfair_em_thining_factor,
        unknown_label=UNKNOWN_LABEL,
    )
    z2 = threshold_sfair(
        img=img,
        k=1.66,
        alpha=sfair_alpha,
        n=sfair_window_size,
        max_iter=sfair_em_max_iter,
        em_thining_factor=sfair_em_thining_factor,
        unknown_label=UNKNOWN_LABEL,
    )

    # Step 2 - Merging: combine the two S-FAIR results
    I_m = np.max([z1, z2], axis=0)  # values in {0, 0.5, 1}

    # Step 3 - Post-Filtering process
    # Step 3.a. - remove stains connected components only surrounded by unknown pixels
    I_m = remove_stains(arr=I_m, stain_max_pixels=stain_max_pixels)

    # Step 3.b. - correct misclassified text pixels
    if postprocess_enabled:
        I_m = correct_misclassified_text_pixels(
            img=img,
            I_m=I_m,
            n=postprocess_window_size,
            sfair_max_iter=postprocess_em_max_iter,
            fair_max_iter=postprocess_max_iter,
            unknown_label=UNKNOWN_LABEL,
        )

    # Step 4. - Final labeling
    I_m = final_labeling(I_m=I_m, unknown_label=UNKNOWN_LABEL, beta=beta)

    I_m = (1 - I_m).astype(np.uint8)  # reverse so that 0 is text and 1 is background

    return I_m
