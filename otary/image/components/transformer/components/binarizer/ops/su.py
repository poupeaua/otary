"""
Official citation:
Su, Bolan & Lu, Shijian & Tan, Chew Lim. (2010).
Binarization of historical document images using the local maximum and minimum.
International Workshop on Document Analysis Systems. 159-166. 10.1145/1815330.1815351.

From:
https://www.researchgate.net/publication/220933012
"""

import numpy as np
from numpy.typing import NDArray

from otary.image.utils.local import high_contrast_local, sum_local


def threshold_su(
    img: NDArray,
    window_size: int = 3,
    n_min: int = -1,
) -> NDArray[np.uint8]:
    """Compute the Su local thresholding.

    Paper (2010):
    https://www.researchgate.net/publication/220933012

    Args:
        img (NDArray): input grayscale image
        window_size (int, optional): window size for local computation. Defaults to 3.
        n_min (int, optional): minimum number of high contrast pixels within the
            neighborhood window. Defaults to -1 meaning that n_min = window_size.

    Returns:
        NDArray[np.uint8]: output thresholded image
    """
    if n_min < 0:
        n_min = window_size

    i_c = high_contrast_local(img=img, window_size=window_size)

    # number of high contrast pixels
    n_e = sum_local(img=i_c.astype(np.float32) / 255, window_size=window_size) + 1e-9

    tmp = (i_c == 255) * img
    img_sum = sum_local(img=tmp, window_size=window_size)
    e_mean = img_sum / n_e

    e_std = np.sqrt((img_sum - n_e * e_mean) ** 2 / 2)

    cond = (n_e >= n_min) & (img <= e_mean + e_std / 2)
    return np.where(cond, 0, 255).astype(np.uint8)
