""" """

import numpy as np
from numpy.typing import NDArray
import cv2

from otary.image.components.transformer.components.binarizer.utils.fair.expectation_maximization import (
    expectation_maximization,
)
from otary.image.components.transformer.components.binarizer.utils.fair.constant import (
    UNKNOWN_LABEL,
)


def dilate_binary_cross(mask: NDArray, distance: int = 1) -> NDArray:
    """Dilate the binary image with a cross-shaped kernel of size
    (2 * distance + 1) x (2 * distance + 1).

    Refers to the FAIR paper to compute the diamond shape extension D_5.

    Args:
        mask (NDArray): input binary array of shape (N, M) with 0 and 1 values.
        distance (int, optional): distance of the cross. Defaults to 1.

    Returns:
        NDArray: dilated array
    """
    mask = (mask > 0).astype(np.uint8)

    # cross-shaped kernel (4-connectivity) - called diamond or D_5 in FAIR paper
    _kernel = (2 * distance + 1, 2 * distance + 1)
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_CROSS, ksize=_kernel)

    dilated = cv2.dilate(src=mask, kernel=kernel, iterations=1)
    return dilated


def remove_stains(arr: NDArray, stain_max_pixels: int = 50) -> NDArray:
    """Remove stains in an image.
    Stains are defined as connected components with a small number of pixels
    (less than stain_max_pixels) and surrounded by unknown labeled pixels.

    Args:
        arr (NDArray): input array image
        stain_max_pixels (int, optional): maximum number of pixels for a stain.
            Defaults to 50.

    Returns:
        NDArray: image without stains
    """
    mask = (arr != UNKNOWN_LABEL).astype(np.uint8)
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
        if np.all(arr[border] == UNKNOWN_LABEL):
            arr[component_mask.astype(bool)] = UNKNOWN_LABEL

    return arr


def correct_misclassified_text_pixels(
    I_m: NDArray,
    img: NDArray,
    n: int,
    max_iter: int,
    em_max_iter: int,
) -> NDArray:
    """Correct potentially misclassified text pixels.
    This is a postprocessing step in the FAIR algorithm.

    Args:
        I_m (NDArray): output of the S-FAIR algorithm (3-values-binarized image)
        img (NDArray): original input image
        n (int): window size
        max_iter (int): maximum number of iterations to try to correct misclassified
            pixels
        em_max_iter (int): Expectation-Maximation (EM)algorithm maximum number of
            iterations

    Returns:
        NDArray: return a corrected 3-values-binarized image
    """
    pad = n // 2
    z_pti_prev = np.zeros_like(I_m)
    for i in range(max_iter):
        z_ti = np.where(I_m == 1, 1, 0)
        z_ui = np.where(I_m == UNKNOWN_LABEL, 1, 0)
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
        gamma = expectation_maximization(x=x, max_iter=em_max_iter)
        centers = gamma[:, pad, pad]
        z = np.where(centers > 0.5, 1, 0)
        I_m[s[:, 0], s[:, 1]] = z

        z_pti_prev = z_pti.copy()

    return I_m


def final_labeling(I_m: NDArray, beta: float = 1.0) -> NDArray:
    """Final labeling postprocessing step for the FAIR algorithm.
    Transform the 3-values-binarized image into a 2-values-binarized image which is
    the typical binarized image.

    Args:
        I_m (NDArray): 3-values-binarized image
        beta (float, optional): factor to define if the unkown pixels
            should be set as text or background. If beta is 1 then
            unknown pixels are set to text if the number of surrounding text pixels
            (N_t) is higher than the number of surrounding background pixels (N_b).
            Simply N_t > N_b. Beta is the value to put more flexibility on the rule
            and thus set unknown pixels to text if N_t > beta * N_b
            Defaults to 1.0.

    Returns:
        NDArray: binarized image
    """
    mask = (I_m == UNKNOWN_LABEL).astype(np.uint8)
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
