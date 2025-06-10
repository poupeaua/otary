"""
Tools function to be used in the Geometry classes
"""

import math

import numpy as np
from numpy.typing import NDArray


def rotate_2d_points(
    points: NDArray,
    angle: float,
    pivot: NDArray,
    is_degree: bool = False,
    is_clockwise: bool = True,
) -> NDArray:
    """Rotate the points.
    A pivot point can be passed as an argument to rotate the object around the pivot

    Args:
        points (NDArray): points to be rotated
        angle (float): rotation angle
        pivot (NDArray): pivot point.
        is_degree (bool, optional): whether the angle is in degree or radian.
            Defaults to False which means radians.
        is_clockwise (bool, optional): whether the rotation is clockwise or
            counter-clockwise. Defaults to True.

    Returns:
        NDArray: rotated points.
    """

    if is_degree:  # transform angle to radian if in degree
        angle = np.deg2rad(angle)

    if not is_clockwise:
        angle = -angle

    # Translate the point so that the pivot is at the origin
    translated_points = points - pivot

    # Define the rotation matrix
    rotation_matrix = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )

    # Apply the rotation matrix to translated point
    rotated_points = np.matmul(translated_points, rotation_matrix.T)

    # Translate the point back to its original space and return
    final_points = rotated_points + pivot
    return final_points
