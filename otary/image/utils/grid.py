"""
Grid-based computations.

Local computation are done using a window that is shifted by 1 pixel in each
direction.

Grid-based computations are more efficient than local computations.
They are done using a grid of size w and a shift of w.

So for a (height, width) grayscale image with a grid of size (w, w), the number of
computations is N = height * width / w**2. Then
"""

from typing import Optional

import numpy as np
from numpy.typing import NDArray


def grid_view(arr: np.ndarray, grid_size: int, pad_value=0):
    """
    Splits a 2D array into non-overlapping (grid x grid) blocks with padding.

    Args:
        arr (np.ndarray): Input 2D array of shape (h, w).
        grid_size (int): Size of each block (grid x grid).
        pad_value (scalar): Value used to pad if array shape is not divisible by grid.

    Returns:
        np.ndarray: View of shape (h//grid, w//grid, grid, grid).
    """
    h, w = arr.shape

    # compute padding needed
    pad_h = (grid_size - (h % grid_size)) % grid_size
    pad_w = (grid_size - (w % grid_size)) % grid_size

    # pad array
    arr_padded = np.pad(
        arr, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=pad_value
    )

    h, w = arr_padded.shape

    # reshape and swap axes to group blocks
    # pylint: disable=too-many-function-args
    blocks = arr_padded.reshape(
        h // grid_size, grid_size, w // grid_size, grid_size
    ).swapaxes(
        1, 2
    )  # shape → (num_blocks_h, num_blocks_w, grid, grid)

    return blocks


def ungrid(
    blocks: NDArray, grid_size: int, original_shape: Optional[tuple[int, int]] = None
) -> NDArray:
    """
    Reconstructs the original 2D array from grid blocks, with options to crop or resize
    to a target shape.

    Args:
        blocks (np.ndarray): Array of shape (num_blocks_h, num_blocks_w, grid, grid)
            containing grid blocks.
        grid_size (int): Size of each grid block.
        reset_original_shape (bool, optional): If True and `original_shape` is
            provided, crops the result to the original shape. Defaults to True.
        original_shape (tuple, optional): The original (height, width) of the array
            before padding.

    Returns:
        np.ndarray: The reconstructed 2D array.
    """
    num_blocks_h, num_blocks_w, _, _ = blocks.shape

    # swap axes back to (num_blocks_h, grid, num_blocks_w, grid)
    arr = blocks.swapaxes(2, 1)

    # reshape to (H, W)
    arr = arr.reshape(num_blocks_h * grid_size, num_blocks_w * grid_size)

    # crop to original shape if requested
    if original_shape is not None:
        h, w = original_shape
        arr = arr[:h, :w]

    return arr
