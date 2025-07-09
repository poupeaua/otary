"""
Thresholding techniques
"""

import cv2
import numpy as np
from numpy.typing import NDArray


def threshold_niblack_like(
    img: np.ndarray,
    method: str = "sauvola",
    window_size: int = 15,
    k: float = 0.5,
    r: float = 128.0,
) -> tuple[NDArray, NDArray[np.uint8]]:
    """Fast implementation of the Niblack-like thresholding.

    It includes the version of the Sauvola thresholding that generally
    gives the best results.

    See https://scikit-image.org/docs/0.24.x/auto_examples/segmentation/\
        plot_niblack_sauvola.html
    for more information about those thresholding methods.
    Function inspired by https://github.com/opencv/opencv_contrib/blob/4.x/modules/\
        ximgproc/src/niblack_thresholding.cpp.

    Originally, the sauvola thresholding was invented for text recognition.

    Args:
        img (np.ndarray): image inputs
        method (str, optional): method to apply.
            Must be in ["niblack", "sauvola", "nick", "wolf"]. Defaults to "sauvola".
        window_size (int, optional): window size. Defaults to 25.
        k (float, optional): k factor. Defaults to 0.5.
        r (float, optional): r value used only in sauvola. Defaults to 128.0.

    Returns:
        tuple[NDArray, NDArray[np.uint8]]: thresh and thresholded image
    """
    # pylint: disable=too-many-locals
    # the window size must be odd and cannot be bigger than the image size
    window_size = min(window_size, img.shape[0], img.shape[1])
    if window_size % 2 == 0:
        window_size -= 1  # Ensure odd

    img = img.astype(np.float32)
    half_win = window_size // 2

    integral_img = cv2.integral(img, sdepth=cv2.CV_64F)
    integral_sqimg = cv2.integral(img**2, sdepth=cv2.CV_64F)

    area = window_size**2

    sum_img = (
        integral_img[window_size:, window_size:]
        - integral_img[:-window_size, window_size:]
        - integral_img[window_size:, :-window_size]
        + integral_img[:-window_size, :-window_size]
    )

    sum_sqimg = (
        integral_sqimg[window_size:, window_size:]
        - integral_sqimg[:-window_size, window_size:]
        - integral_sqimg[window_size:, :-window_size]
        + integral_sqimg[:-window_size, :-window_size]
    )

    mean = sum_img / area
    var = (sum_sqimg - (sum_img**2) / area) / area
    std = np.sqrt(var)

    if method == "sauvola":
        thresh = mean * (1 + k * ((std / r) - 1))
    elif method == "niblack":
        thresh = mean + k * std
    elif method == "wolf":
        max_std = np.max([std, 1e-5])  # Avoid division by zero
        min_i = np.min(img)
        thresh = mean + k * (std / max_std) * (mean - min_i)
    elif method == "nick":
        thresh = mean + k * np.sqrt(var + mean**2)
    else:
        raise ValueError(f"Unknown method {method} for threshold_niblack_like")

    thresh_full = np.pad(thresh, half_win, mode="edge")
    img_thresholded = (img > thresh_full).astype(np.uint8) * 255

    return thresh_full, img_thresholded
