"""
Tools function to be used
"""

import numpy as np


def assert_transform_shift_vector(vector: np.ndarray) -> np.ndarray:
    """Assert and transforms shift vector.
    Returns the input shift vector as a single 2D point which would represent the
    vector as if its first point was the origin (0, 0).

    Args:
        vector (np.ndarray): shift vector

    Returns:
        np.ndarray: validated vector
    """
    vector = np.asarray(vector)
    if vector.shape == (2, 2):
        vector = vector[1] - vector[0]  # set the vector to be defined as one point
    if vector.shape != (2,):
        raise ValueError("The input vector {vector} does not have the expected shape.")
    return vector
