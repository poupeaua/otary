"""
Computing a Background Surface Estimation (BSE) is a way of locally estimating the
local average background in order to correct the foreground by this local value
everywhere in the image.

This can be useful as an intermediate step for several binarization methods like
Gatos or AdOtsu (Adaptive Otsu Thresholding).

Generally, the Background Estimation is computed thanks to two elements:
1. the input image
2. a rough binarization of the input image.

The rough binarization is usually computed with a method like Otsu or Sauvola since
those methods are considered as classics for binarization.
"""

from typing import Optional
import numpy as np
from numpy.typing import NDArray

from otary.image.utils.grid import grid_view, ungrid
from otary.image.utils.local import sum_local, windowed_convsum


def bse_checks(img: NDArray, binary: NDArray) -> None:
    """Background Surface Estimation check of input images

    Args:
        img (NDArray): input image in [0, 255]
        binary (NDArray): a rough binarization in [0, 1]. 0 are the background
            and 1 are the foreground. This is just a binarized image with values
            of only 0 or 255 divided by 255.
    """
    max_img, min_img = np.max(img), np.min(img)
    if not (0 <= max_img <= 255) or not (0 <= min_img <= 255) or img.dtype != np.uint8:
        raise ValueError("Input image must be in [0, 255] and of type uint8")

    unique_bin = np.unique(binary)
    if len(unique_bin) != 2 or 0 not in unique_bin or 1 not in unique_bin:
        raise ValueError("Input image must be a binary image with value in [0, 1].")


# ------------------------------------- GATOS -----------------------------------------


def background_surface_estimation_gatos(
    img: NDArray, binary: NDArray, window_size: int
) -> NDArray[np.uint8]:
    r"""Background Surface Estimation using the Gatos paper called:
    "Adaptive degraded document image binarization".

    This background estimation is maybe the most natural one. It first computes the
    a rough binarization of the input image. It updates the pixels of the input image
    where the rough binarization is 0 (foreground, meaning probably text) by an
    local average around each pixels using only the values of the pixels that are
    1 (background). The pixels where the rough binarization is 1 (background) remain
    unchanged.

    Here is the mathematical formula:

    $$
    B(x,y) = \left\{ \begin{array}{cl}
    I(x,y) & if \ S(x,y)) = 1 \\
    \dfrac{\sum_{x_i = x - dx}^{x+dx}\sum_{y_i = y - dy}^{y+dy}I(x_i, y_i) S(x_i, y_i)}
    {\sum_{x_i = x - dx}^{x+dx}\sum_{y_i = y - dy}^{y+dy}S(x_i, y_i)} & if \ S(x,y) = 0
    \end{array} \right.
    $$

    Args:
        img (NDArray): input image in [0, 255]
        binary (NDArray): a rough binarization in [0, 1]. 0 are the background
            and 1 are the foreground. This is just a binarized image with values
            of only 0 or 255 divided by 255.
        window_size (int): window size to perform the background estimation

    Returns:
        NDArray[np.uint8]: background estimation image
    """
    bse_checks(img=img, binary=binary)

    # compute the background surface estimation
    bse = windowed_convsum(img1=img, img2=binary, window_size=window_size) / (
        sum_local(img=binary, window_size=window_size) + 1e-9
    )

    # for pixels that were rough binarized as background use the value from input image
    bse_corrected = np.where(
        binary == 1,
        img,  # when is background
        bse,  # when is foreground
    )

    return bse_corrected


# ------------------------------------- AdOtsu ----------------------------------------


def background_surface_estimation_adotsu(
    img: NDArray,
    binary: NDArray,
    grid_size_init: int,
    grid_size_min: Optional[int] = None,
) -> NDArray[np.uint8]:
    """Compute the Background Surface Estimation described in the AdOtsu paper

    Args:
        img (NDArray): _description_
        binary (NDArray): _description_
        grid_size_init (int): _description_
        grid_size_min (Optional[int], optional): _description_. Defaults to None.

    Returns:
        NDArray[np.uint8]: _description_
    """
    bse_checks(img=img, binary=binary)

    if grid_size_min is None:
        grid_size_min = np.sqrt(img.shape[0] * img.shape[1]) // 10

    # pre-compute the scales
    cur_scale = grid_size_init
    scales = []
    while cur_scale > grid_size_min:
        cur_scale = cur_scale // 2
        scales.append(cur_scale)

    # Grayscale Background Level - first scale easy computation
    gbl = np.full(shape=img.shape, fill_value=np.mean(img[binary == 1]), dtype=np.uint8)

    # compute the the Grayscale Background Level for the other scales
    gbls = [gbl]
    for scale in scales:
        # get the patched images
        img_ = grid_view(arr=img, grid_size=scale)
        binary_ = grid_view(arr=binary, grid_size=scale)

        # Compute mean background value - same methodology as in Gatos
        sum_vals = np.sum(img_ * binary_, axis=(2, 3))
        count_vals = np.sum(binary_, axis=(2, 3))
        mean_vals = np.divide(sum_vals, count_vals, where=count_vals > 0)

        # bring back to the original image shape
        mean_vals = np.repeat(
            np.repeat(mean_vals[:, :, None, None], scale, axis=2), scale, axis=3
        )
        mean_vals = ungrid(
            blocks=mean_vals,
            grid_size=scale,
            original_shape=(img.shape[0], img.shape[1]),
        ).astype(np.uint8)

        gbls.append(mean_vals)

    # use the darkest value in background
    bse = np.min(np.asarray(gbls), axis=0)

    bse = np.where(
        binary == 1,
        img,  # when is background
        bse,  # when is foreground
    )

    return bse
