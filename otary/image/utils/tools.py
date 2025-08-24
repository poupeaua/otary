"""
General utility functions for image processing.
"""

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
