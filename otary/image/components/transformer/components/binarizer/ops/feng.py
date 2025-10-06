"""
Official Citation:
Feng, Mengling & Tan, Yap-Peng. (2004).
Contrast adaptive binarization of low quality document images.
IEICE Electronic Express. 1. 501-506. 10.1587/elex.1.501.

From:
https://www.researchgate.net/publication/220305658
"""

import numpy as np
from numpy.typing import NDArray

from otary.image.utils.local import mean_local, min_local, variance_local


def threshold_feng(
    img: NDArray,
    w1: int = 19,
    w2: int = 33,
    alpha1: float = 0.12,
    k1: float = 0.25,
    k2: float = 0.04,
    gamma: float = 2.0,
) -> NDArray[np.uint8]:
    """Implementation of the Feng thresholding method.

    Paper (2004):
    https://www.jstage.jst.go.jp/article/elex/1/16/1_16_501/_pdf

    Args:
        img (NDArray): input grayscale image
        w1 (int, optional): primary window size. Defaults to 19.
        w2 (int, optional): secondary window value. Defaults to 33.
        alpha1 (float, optional): alpha1 value. Defaults to 0.12.
        k1 (float, optional): k1 value. Defaults to 0.25.
        k2 (float, optional): k2 value. Defaults to 0.04.
        gamma (float, optional): gamma value. Defaults to 2.0.

    Returns:
        NDArray[np.uint8]: output thresholded image
    """
    # pylint: disable=too-many-arguments, too-many-positional-arguments
    # pylint: disable=too-many-locals
    if not 0 < w1 < w2:
        raise ValueError("Using Feng thresholding requires 0 < w1 < w2")

    # mean local on primary window
    m = mean_local(img=img, window_size=w1)

    # min local on primary window
    M = min_local(img=img, window_size=w1)

    # std local on primary window
    sqmean = mean_local(img=img**2, window_size=w1)
    var = sqmean - m**2
    s = np.sqrt(np.clip(var, 0, None))

    # std in local secondary window
    r_s = variance_local(img=img, window_size=w2)

    # setup parameters
    normalized_std = s / (r_s + 1e-9)
    alpha2 = k1 * (normalized_std**gamma)
    alpha3 = k2 * (normalized_std**gamma)

    # compute threshold
    thresh = (1 - alpha1) * m + alpha2 * normalized_std * (m - M) + alpha3 * M
    thresh = np.clip(thresh, 0, 255)

    img_thresholded = (img > thresh).astype(np.uint8) * 255
    return img_thresholded
