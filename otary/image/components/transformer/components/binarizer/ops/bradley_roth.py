"""
Official citation:
Bradley, Derek & Roth, Gerhard. (2007).
Adaptive Thresholding using the Integral Image.
J. Graphics Tools. 12. 13-21. 10.1080/2151237X.2007.10129236.

From:
https://www.researchgate.net/publication/220494200
"""

from numpy.typing import NDArray
import numpy as np

from otary.image.utils.local import mean_local


def threshold_bradley(
    img: NDArray, window_size: int = 15, t: float = 0.15
) -> NDArray[np.uint8]:
    """Implementation of the Bradley & Roth thresholding method.
    This is actually a very easy thresholding method solely depending on the
    local mean.

    This is a local thresholding method that computes the threshold for a pixel
    based on a small region around it.

    Args:
        img (NDArray): input image
        window_size (int, optional): window size for local computations.
            Defaults to 15.
        t (float, optional): t value in [0, 1]. Defaults to 0.15.

    Returns:
        NDArray[np.uint8]: output thresholded image
    """
    if not 0 < t < 1:
        raise ValueError("t must be in range ]0, 1[")

    m = mean_local(img=img, window_size=window_size)
    img_thresholded = (img > m * (1 - t)).astype(np.uint8) * 255
    return img_thresholded
