"""
Farrahi Moghaddam, Reza & Cheriet, Mohamed. (2012).
AdOtsu: An adaptive and parameterless generalization of Otsu's method for document
image binarization.
Pattern Recognition. 45. 2419-2431. 10.1016/j.patcog.2011.12.013.

From:
https://www.researchgate.net/publication/220602345
"""

import cv2
import numpy as np
from numpy.typing import NDArray

from otary.image.components.transformer.components.binarizer.ops.niblack_like import (
    threshold_niblack_like,
)
from otary.image.components.transformer.components.binarizer.ops.otsu import (
    otsu_vectorized,
)
from otary.image.utils.background import background_surface_estimation_adotsu
from otary.image.utils.grid import grid_view, ungrid


def threshold_adotsu(
    img: NDArray,
    grid_size: int = 50,
    k_sigma: float = 1.6,
    n_steps: int = 2,
    is_multiscale_enabled: bool = False,
) -> NDArray[np.uint8]:
    r"""Implementation of the AdOtsu thresholding method.

    AdOtsu is computed this way on a grid-based approach:

    $$
    T_{AdOtsu, u}(x) = \epsilon + (Otsu(x) + \epsilon) \times \Theta(\sigma(x) - k_{\sigma}\sigma_{EB}(x)) # pylint: disable=line-too-long
    $$

    Currently the Multi-Scale AdOtsu is not implemented.
    It only computes using the basic AdOtsu pipeline.
    Still the Background Surface Estimation is computed using a multiscale approach
    as described in the paper.

    Args:
        img (NDArray): input image
        grid_size (int, optional): window size for local computations.
            Defaults to 15.
        k_sigma (float, optional): k_sigma value in [1, 2]. Defaults to 1.6.
        n_steps (int, optional): number of iterations to update the binarization by
            estimating a new background surface. Defaults to 2.
        is_multiscale_enabled (bool, optional): is multiscale enabled.
            Defaults to False.

    Returns:
        NDArray[np.uint8]: output thresholded image
    """
    if is_multiscale_enabled:
        raise NotImplementedError("Multiscale is not implemented for AdOtsu")

    # produce the rough binarization
    binary = threshold_niblack_like(img=img, method="sauvola", k=0.1)[1]
    binary_bool = cv2.erode(binary, np.ones((3, 3)), iterations=4) / 255

    for i in range(n_steps):  # update the binarized image to a more accurate one
        binary = adotsu_single_step(
            img=img, binary=binary_bool, grid_size=grid_size, k_sigma=k_sigma
        )

        if i != n_steps - 1:  # while not last step turn the binary into bool
            binary_bool = binary / 255

    return binary


def adotsu_single_step(
    img: NDArray, binary: NDArray, grid_size: int, k_sigma: float
) -> NDArray[np.uint8]:
    """Procedure for a single step of the AdOtsu thresholding

    Args:
        img (NDArray): input image
        binary (NDArray): current binary image
        grid_size (int): grid size since the AdOtsu is computed on a grid-based approach
        k_sigma (float): k sigma value from the base AdOtsu algorithm

    Returns:
        NDArray[np.uint8]: new binary image
    """
    # pylint: disable=too-many-locals
    bse = background_surface_estimation_adotsu(
        img=img, binary=binary, grid_size_init=grid_size
    )

    # grid based view as patches
    pad = -1
    img_grid = grid_view(arr=img, grid_size=grid_size, pad_value=pad)
    bse_grid = grid_view(arr=bse, grid_size=grid_size, pad_value=pad)

    # otsu threshold in each patch
    otsu = otsu_vectorized(patches=img_grid, excluding_padding_values=pad)

    # variance calculation for each patch in both img and bse
    var_img = np.var(img_grid, axis=(2, 3), where=img_grid != pad)
    var_bse = np.var(bse_grid, axis=(2, 3), where=bse_grid != pad)
    step_fn = np.where(var_img > k_sigma * var_bse, 1, 0)

    # AdOtsu threshold
    eps = 1e-9
    adotsu_t = eps + (otsu + eps) * step_fn

    # bring back to the original image shape
    tmp = np.repeat(
        np.repeat(adotsu_t[:, :, None, None], grid_size, axis=2), grid_size, axis=3
    )
    thresh = ungrid(
        blocks=tmp,
        grid_size=grid_size,
        original_shape=(img.shape[0], img.shape[1]),
    )

    # intermediate step for threshold value interpolation after grid-based approach
    grid_size_odd = grid_size + 1 if grid_size % 2 == 0 else grid_size
    thresh = cv2.GaussianBlur(thresh, (grid_size_odd, grid_size_odd), 0)

    img_thresholded = (img > thresh).astype(np.uint8) * 255
    return img_thresholded
