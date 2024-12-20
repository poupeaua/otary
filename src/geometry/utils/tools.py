"""
Tools function to be used in the Geometry classes
"""

import numpy as np


def validate_shift_vector(vector: np.ndarray) -> np.ndarray:
    """Validate the shift vector before executing operation

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
