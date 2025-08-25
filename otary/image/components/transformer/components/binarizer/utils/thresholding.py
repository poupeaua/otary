"""
Thresholding techniques
"""

import numpy as np
from numpy.typing import NDArray

from otary.image.utils.intensity import intensity_local_v2, max_local
from otary.image.utils.tools import check_transform_window_size


def threshold_niblack_like(
    img: np.ndarray,
    method: str = "sauvola",
    window_size: int = 15,
    k: float = 0.5,
    r: float = 128.0,
) -> tuple[NDArray, NDArray[np.uint8]]:
    """Fast implementation of the Niblack-like thresholdings.
    These thresholdings are similar so we just put them in the same utils function.

    These thresholding methods are local thresholding methods. This means that
    the threshold value is computed for each pixel based on the pixel values
    in a window around the pixel. The window size is a parameter of the function.

    Local thresholding methods are generally better than global thresholding
    methods (like Otsu or adaptive thresholding) for images with varying
    illumination.

    It includes the following methods:
    - Niblack
    - Sauvola
    - Wolf
    - Nick

    See https://scikit-image.org/docs/0.24.x/auto_examples/segmentation/\
        plot_niblack_sauvola.html
    for more information about those thresholding methods.

    Originally, the sauvola thresholding was invented for text recognition like
    most of the niblack-like thresholding methods.

    Args:
        img (np.ndarray): image inputs
        method (str, optional): method to apply.
            Must be in ["niblack", "sauvola", "nick", "wolf"]. Defaults to "sauvola".
        window_size (int, optional): window size. Defaults to 15.
        k (float, optional): k factor. Defaults to 0.5.
        r (float, optional): r value used only in sauvola. Defaults to 128.0.

    Returns:
        tuple[NDArray, NDArray[np.uint8]]: thresh and thresholded image
    """
    window_size = check_transform_window_size(img, window_size)

    img = img.astype(np.float32)

    # compute intensity representation of image
    mean = intensity_local_v2(img=img, window_size=window_size)
    sqmean = intensity_local_v2(img=img**2, window_size=window_size, normalize=True)
    var = sqmean - mean**2
    std = np.sqrt(np.clip(var, 0, None))

    # compute the threshold matrix
    if method == "sauvola":
        thresh = mean * (1 + k * ((std / r) - 1))
    elif method == "wan":
        wan_mean = (max_local(img=img, window_size=window_size) + mean) / 2
        thresh = wan_mean * (1 + k * ((std / r) - 1))
    elif method == "niblack":
        thresh = mean + k * std
    elif method == "wolf":
        max_std = np.max([std, 1e-5])  # Avoid division by zero
        min_img = np.min(img)
        thresh = mean - k * (mean - min_img - std * (mean - min_img) / max_std)
    elif method == "nick":
        thresh = mean + k * np.sqrt(var + sqmean)
    else:
        raise ValueError(f"Unknown method {method} for threshold_niblack_like")

    # compute the output, meaning the threshold and the thresholded image
    img_thresholded = (img > thresh).astype(np.uint8) * 255

    return thresh, img_thresholded
