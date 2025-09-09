"""
Efficient local computations for images.
Local computations are all kind of computations done using a local window kind
of like convolution.
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
        img (NDArray): input grayscale image
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
        img (NDArray): input grayscale image
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


def mean_local(
    img: NDArray, window_size: int = 15, border_type: int = cv2.BORDER_DEFAULT
) -> NDArray:
    """Compute the local mean of the image.
    The local mean representation is the mean of the pixel values in a
    window of size (window_size, window_size) around each pixel.

    This is essentially computing the intensity without normalizing over the
    window area.

    Args:
        img (NDArray): input grayscale image
        window_size (int, optional): window size. Defaults to 15.
        border_type (int, optional): border type to use for the integral image.
            Defaults to cv2.BORDER_DEFAULT.
        cast_uint8 (bool, optional): whether to cast the intensity to integer.
            Defaults to False.

    Returns:
        NDArray: local sum representation of the image
    """
    return intensity_local_v2(
        img=img,
        window_size=window_size,
        border_type=border_type,
        normalize=True,
        cast_uint8=False,
    )


def sum_local(
    img: NDArray, window_size: int = 15, border_type: int = cv2.BORDER_DEFAULT
) -> NDArray:
    """Compute the local sum of the image.
    The local sum representation is the sum of the pixel values in a
    window of size (window_size, window_size) around each pixel.

    The result is not of dtype np.uint8 since the sum can exceed very quickly 255
    for a single pixel zone.

    This is essentially computing the intensity without normalizing over the
    window area.

    Args:
        img (NDArray): input grayscale image
        window_size (int, optional): window size. Defaults to 15.
        border_type (int, optional): border type to use for the integral image.
            Defaults to cv2.BORDER_DEFAULT.
        cast_uint8 (bool, optional): whether to cast the intensity to integer.
            Defaults to False.

    Returns:
        NDArray: local sum representation of the image
    """
    return intensity_local_v2(
        img=img,
        window_size=window_size,
        border_type=border_type,
        normalize=False,
        cast_uint8=False,
    )


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
        img (NDArray): input grayscale image
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
    else:
        img_max = img_max.astype(np.float32)

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
        img (NDArray): input grayscale image
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
    else:
        img_min = img_min.astype(np.float32)

    return img_min


def contrast_local(
    img: NDArray, window_size: int, eps: float = 1e-9
) -> NDArray[np.uint8]:
    """Compute the contrast image

    Args:
        img (NDArray): input grayscale image
        window_size (int): window size for local computations
        eps (float, optional): epsilon value to avoid division by zero.
            Defaults to 1e-9.

    Returns:
        NDArray[np.uint8]: contrast representation of the image
    """
    min_ = min_local(img=img, window_size=window_size)
    max_ = max_local(img=img, window_size=window_size)
    contrast_ = (max_ - min_) / (max_ + min_ + eps) * 255
    return contrast_.astype(np.uint8)


def high_contrast_local(img: NDArray, window_size: int) -> NDArray[np.uint8]:
    """Compute the high contrast image.

    Simply performs in order the following operations:
    - contrast local
    - Otsu thresholding

    Useful for several binarization methods like Su or ISauvola.

    Args:
        img (NDArray): input grayscale image
        window_size (int): window size for local computations

    Returns:
        NDArray[np.uint8]: high contrast representation of the image
    """
    img_ = contrast_local(img=img, window_size=window_size)
    _, img_otsu = cv2.threshold(img_, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img_otsu.astype(np.uint8)


def wiener_filter(img: NDArray, window_size: int = 3) -> NDArray[np.uint8]:
    """Compute the Wiener filter from the Gatos paper.

    https://users.iit.demokritos.gr/~bgat/PatRec2006.pdf

    Args:
        img (NDArray): input image
        window_size (int): window size for local computations. Default to 3.

    Returns:
        NDArray[np.uint8]: image with Wiener filter applied
    """
    mean = mean_local(img=img, window_size=window_size)
    sqmean = mean_local(img=img**2, window_size=window_size)
    var = sqmean - mean**2
    avg_var = mean_local(img=var, window_size=window_size)  # mean of the variance
    wiener = mean + (var - avg_var) * (img - mean) / (var + 1e-9)
    return wiener


def windowed_convsum(img1: NDArray, img2: NDArray, window_size: int) -> NDArray:
    """Compute the windowed multiplication of two images.
    It is essentially computing the convolution sum of the two images over a
    sliding window.

    Args:
        img1 (NDArray): first image
        img2 (NDArray): second image
        window_size (int): window size for local computations.

    Returns:
        NDArray: windowed multiplication of the two images
    """
    prod = img1 * img2  # standard Hadamard multiplication
    result = sum_local(img=prod, window_size=window_size)
    return result
