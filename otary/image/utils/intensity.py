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
    cast_uint8: bool = False,
) -> NDArray:
    """Compute the local intensity of the image.
    The intensity representation is the sum of the pixel values in a
    window of size (window_size, window_size) around each pixel.

    This function makes the whole computation using the integral image low-level method.
    This way one can really understand how the intensity calculation is done.

    Args:
        img (NDArray): input image
        window_size (int, optional): window size. Defaults to 15.
        border_type (int, optional): border type to use for the integral image.
            Defaults to cv2.BORDER_DEFAULT.
        normalize (bool, optional): whether to normalize the intensity by the
            area of the window. Defaults to True.
        cast_uint8 (bool, optional): whether to cast the intensity to integer.
            Defaults to False.

    Returns:
        NDArray: intensity representation of the image
    """
    w = check_transform_window_size(img=img, window_size=window_size)

    im = img.astype(np.float32)

    half_w = w // 2
    img_withborders = cv2.copyMakeBorder(
        im, half_w, half_w, half_w, half_w, borderType=border_type
    )

    _img = cv2.integral(img_withborders, sdepth=cv2.CV_32F)

    img_intensity = _img[w:, w:] - _img[:-w, w:] - _img[w:, :-w] + _img[:-w, :-w]

    if normalize:
        img_intensity = img_intensity / (w**2)

    if cast_uint8:
        img_intensity = np.clip(np.round(img_intensity), 0, 255).astype(np.uint8)

    return img_intensity


def intensity_local_v2(
    img: NDArray,
    window_size: int = 15,
    border_type: int = cv2.BORDER_DEFAULT,
    normalize: bool = True,
    cast_uint8: bool = False,
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
        cast_uint8 (bool, optional): whether to cast the intensity to integer.
            Defaults to False.

    Returns:
        NDArray: intensity representation of the image
    """
    w = check_transform_window_size(img=img, window_size=window_size)

    ddepth = -1 if cast_uint8 else cv2.CV_32F

    img_intensity = cv2.boxFilter(
        img, ddepth=ddepth, ksize=(w, w), normalize=normalize, borderType=border_type
    )

    return img_intensity


def max_local(
    img: NDArray,
    window_size: int = 15,
    border_type: int = cv2.BORDER_DEFAULT,
    cast_uint8: bool = False,
) -> NDArray:
    """Compute the local maximum of the image.
    The local maximum representation is the maximum pixel value in a
    window of size (window_size, window_size) around each pixel.

    Args:
        img (NDArray): input image
        window_size (int, optional): window size. Defaults to 15.
        border_type (int, optional): border type to use for the integral image.
            Defaults to cv2.BORDER_DEFAULT.
        cast_uint8 (bool, optional): whether to cast the intensity to integer.
            Defaults to False.

    Returns:
        NDArray: local maximum representation of the image
    """
    w = check_transform_window_size(img=img, window_size=window_size)

    img_max = cv2.dilate(
        img,
        kernel=np.ones((w, w), dtype=np.uint8),
        borderType=border_type,
    )

    if cast_uint8:
        img_max = np.clip(np.round(img_max), 0, 255).astype(np.uint8)

    return img_max

def min_local(
    img: NDArray,
    window_size: int = 15,
    border_type: int = cv2.BORDER_DEFAULT,
    cast_uint8: bool = False,
) -> NDArray:
    """Compute the local minimum of the image.
    The local minimum representation is the minimum pixel value in a
    window of size (window_size, window_size) around each pixel.

    Args:
        img (NDArray): input image
        window_size (int, optional): window size. Defaults to 15.
        border_type (int, optional): border type to use for the integral image.
            Defaults to cv2.BORDER_DEFAULT.
        cast_uint8 (bool, optional): whether to cast the intensity to integer.
            Defaults to False.

    Returns:
        NDArray: local minimum representation of the image
    """
    w = check_transform_window_size(img=img, window_size=window_size)

    img_min = cv2.erode(
        img,
        kernel=np.ones((w, w), dtype=np.uint8),
        borderType=border_type,
    )

    if cast_uint8:    
        img_min = np.clip(np.round(img_min), 0, 255).astype(np.uint8)

    return img_min
