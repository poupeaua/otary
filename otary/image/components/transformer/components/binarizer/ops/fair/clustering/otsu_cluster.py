"""
FAIR clustering using Otsu thresholding
"""

import numpy as np
from numpy.typing import NDArray

from otary.image.components.transformer.components.binarizer.ops.otsu import (
    otsu_vectorized,
)


def otsu_clustering(x: NDArray) -> NDArray:
    """This is not really a clustering algorithm.
    But the idea is to use Otsu thresholding to binarize the image as if it was a
    clustering method.

    Args:
        arr (NDArray): input patches as shape (N, n, n)

    Returns:
        NDArray: output threshold as shape (N, n, n)
    """
    assert x.ndim == 3
    x = x[np.newaxis, ...]  # add one dimension at front
    thresh = otsu_vectorized(patches=x)
    thresh = thresh[..., np.newaxis, np.newaxis]
    output = (x < thresh).astype(np.uint8)  # 0 is background and 1 is foreground
    output = output.squeeze()  # remove one dimension at front
    return output
