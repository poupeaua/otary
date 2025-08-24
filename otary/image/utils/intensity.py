"""
Efficient intensity transformations for images.
"""

import cv2
import numpy as np
from numpy.typing import NDArray

from otary.image.utils.tools import check_transform_window_size


def intensity_local(
    img: NDArray,
    window_size: int = 15,
    border_type: int = cv2.BORDER_DEFAULT,
    normalize: bool = True,
    cast_int: bool = False,
) -> NDArray:
    """Compute the local intensity of the image.
    The intensity representation is the sum of the pixel values in a
    window of size (window_size, window_size) around each pixel.

    This function makes the whole computation using the integral image low-level method.
    This way one can really understand how the intensity calculation is done.

    Plus this method offers the following advantage over the box filter from OpenCV:
    - no clipping of values in [0, 255] in the non-normalized case. Can be
        useful for some applications.


    Args:
        img (NDArray): input image
        window_size (int, optional): window size. Defaults to 15.
        border_type (int, optional): border type to use for the integral image.
            Defaults to cv2.BORDER_DEFAULT.
        normalize (bool, optional): whether to normalize the intensity by the
            area of the window. Defaults to True.

    Returns:
        NDArray: intensity representation of the image
    """
    w = check_transform_window_size(img=img, window_size=window_size)

    im = img.astype(np.float32)
    img_withborders = cv2.copyMakeBorder(im, 1, 1, 1, 1, borderType=border_type)

    _img = cv2.integral(img_withborders, sdepth=cv2.CV_32F)

    img_intensity = _img[w:, w:] - _img[:-w, w:] - _img[w:, :-w] + _img[:-w, :-w]

    if normalize:
        img_intensity = img_intensity / (w**2)

    if cast_int:
        img_intensity = np.clip(np.round(img_intensity), 0, 255).astype(np.uint8)

    return img_intensity


def intensity_local_v2(
    img: NDArray,
    window_size: int = 15,
    border_type: int = cv2.BORDER_DEFAULT,
    normalize: bool = True,
    cast_int: bool = False,
) -> NDArray:
    """Compute the local intensity of the image.
    The intensity representation is the sum of the pixel values in a
    window of size (window_size, window_size) around each pixel.

    This version uses the box filter from OpenCV which is faster is most cases.
    We recommend this version unless you need the advantages of the first version.

    Args:
        img (NDArray): input image
        window_size (int, optional): window size. Defaults to 15.
        border_type (int, optional): border type to use for the integral image.
            Defaults to cv2.BORDER_DEFAULT.
        normalize (bool, optional): whether to normalize the intensity by the
            area of the window. Defaults to True.

    Returns:
        NDArray: intensity representation of the image
    """
    w = check_transform_window_size(img=img, window_size=window_size)

    ddepth = -1 if cast_int else cv2.CV_32F

    img_intensity = cv2.boxFilter(
        img, ddepth=ddepth, ksize=(w, w), normalize=normalize, borderType=border_type
    )

    return img_intensity
