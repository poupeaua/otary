"""
Implementation of Otsu threshold
"""

import numpy as np
from numpy.typing import NDArray


def otsu_vectorized(
    patches: NDArray, nbins: int = 256, excluding_padding_values: int = -1
):
    """
    Compute Otsu threshold for each patch in (num_blocks_h, num_blocks_w, grid, grid).
    Assumes patches are uint8 images.

    This can be seen as a vectorized implementation of the Otsu algorithm for
    faster parralel computation.

    Args:
        patches (NDArray): Array of shape (num_blocks_h, num_blocks_w, grid, grid)
            containing patches.
        nbins (int, optional): Number of bins for histogram computation.
            Which is one for each possible pixel value in [0, 255].
            Defaults to 256.
        excluding_padding_values (int, optional): Value used to exclude from histogram
            computation. Defaults to -1.

    Returns:
        NDArray: Array of shape (num_blocks_h, num_blocks_w) containing Otsu threshold
    """
    # pylint: disable=too-many-locals
    nbh, nbw, gh, gw = patches.shape
    nblocks = nbh * nbw
    flat = patches.reshape(nblocks, gh * gw)

    # Exclude values -1 from histogram computation
    hist = np.zeros((nblocks, nbins), dtype=np.int32)
    for i in range(nblocks):
        vals = flat[i]
        vals = vals[vals != excluding_padding_values]
        if vals.size > 0:
            hist[i] = np.bincount(vals, minlength=nbins)
        else:
            hist[i] = 0  # If all values are excluding_padding_values, histogram is zero

    # normalize to probabilities
    hist = hist.astype(np.float32)
    hist_sum = hist.sum(axis=1, keepdims=True)
    # Avoid division by zero
    hist_sum[hist_sum == 0] = 1
    hist = hist / hist_sum

    # cumulative sums (class probabilities)
    omega = np.cumsum(hist, axis=1)
    mu = np.cumsum(hist * np.arange(nbins), axis=1)

    mu_t = mu[:, -1]  # total mean

    # between-class variance for all thresholds
    sigma_b = (mu_t[:, None] * omega - mu) ** 2 / (omega * (1 - omega) + 1e-10)

    # best threshold = argmax variance
    thresholds = sigma_b.argmax(axis=1).astype(np.uint8)

    return thresholds.reshape(nbh, nbw)
