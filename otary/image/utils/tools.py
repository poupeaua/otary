"""
General utility functions for image processing.
"""

import numpy as np
import cv2

from numpy.typing import NDArray


def check_transform_window_size(img: NDArray, window_size: int) -> int:
    """Ensure the window size must be odd and cannot be bigger than the image size

    Args:
        img (NDArray): input image
        window_size (int): input window size

    Returns:
        int: checked and eventually transformed window size
    """
    window_size = min(window_size, img.shape[0], img.shape[1])  # Ensure <= image size

    window_size = max(3, window_size)  # Ensure >= 3

    if window_size % 2 == 0:
        window_size -= 1  # Ensure odd
    return window_size


def bwareaopen(img: NDArray, n_min_pixels: int, connectivity: int = 8) -> NDArray:
    """
    Remove small connected components.

    Args:
        img (np.ndarray): Binary image (0 or 255).
        n_min_pixels (int): Minimum pixel area of components to keep.
        connectivity (int, optional): Connectivity of the neighborhood. Defaults to 8.

    Returns:
        np.ndarray: Cleaned binary image.
    """
    unique_val = np.unique(img)
    if len(unique_val) != 2 or 0 not in unique_val or 255 not in unique_val:
        raise ValueError("Input image must be a binary image with value in [0, 255].")

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        image=img, connectivity=connectivity
    )

    cleaned = np.zeros_like(img, dtype=np.uint8)
    for i in range(1, num_labels):  # skip background (0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= n_min_pixels:
            cleaned[labels == i] = 255

    return cleaned
