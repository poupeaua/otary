"""
Official Citation:
BERNSEN, J., 1986. Dynamic thresholding of grey-level images.
In: Eighth International Conference on Pattern Recognition.
Proceedings. S. 1251-1255

From:
https://www.tib.eu/en/search/id/tema-archive%3ATEMAE87090699070/Dynamic-thresholding-of-grey-level-images/
"""

from numpy.typing import NDArray
import numpy as np

from otary.image.utils.local import max_local, min_local


def threshold_bernsen(
    img: NDArray,
    window_size: int = 75,
    contrast_limit: float = 25,
    threshold_global: int = 100,
) -> NDArray[np.uint8]:
    """Implementation of the Bernsen thresholding method.

    This is a local thresholding method that computes the threshold for a pixel
    based on a small region around it.

    Args:
        img (NDArray): input image
        window_size (int, optional): window size for local computations.
            Defaults to 75.
        contrast_limit (float, optional): contrast limit. If the
            contrast is higher than this value, the pixel is thresholded by the
            bernsen threshold otherwise the global threshold is used.
            Defaults to 25.
        threshold_global (int, optional): global threshold. Defaults to 100.

    Returns:
        NDArray[np.uint8]: output thresholded image
    """
    z_high = max_local(img=img, window_size=window_size)
    z_low = min_local(img=img, window_size=window_size)
    bernsen_contrast = z_high - z_low
    bernsen_threshold = (z_high + z_low) / 2

    threshold_local = np.where(
        bernsen_contrast > contrast_limit,
        bernsen_threshold,
        threshold_global,  # global threshold is broadcast
    )

    img_thresholded = (img > threshold_local).astype(np.uint8) * 255

    return img_thresholded
