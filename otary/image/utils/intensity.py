"""
Efficient intensity transformations for images.
"""

import cv2
import numpy as np
from numpy.typing import NDArray

from otary.image.utils.tools import check_transform_window_size

def intensity(img: NDArray, window_size: int = 15) -> NDArray:
    """Generate the intensity representation of the image.
    The intensity representation is the sum of the pixel values in a
    window of size (window_size, window_size) around each pixel.

    The resulting array has a shape of (H - window_size + 1, W - window_size + 1)
    where H and W are the height and width of the input image.

    Args:
        img (NDArray): input image
        window_size (int, optional): window size. Defaults to 25.

    Returns:
        NDArray: intensity representation of the image
    """
    window_size = check_transform_window_size(img, window_size)

    img = img.astype(np.float32)

    integral_img = cv2.integral(img, sdepth=cv2.CV_64F)

    img_intensity = (
        integral_img[window_size:, window_size:]
        - integral_img[:-window_size, window_size:]
        - integral_img[window_size:, :-window_size]
        + integral_img[:-window_size, :-window_size]
    )

    return img_intensity
