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


def get_shared_point_indices(
    points_to_check: NDArray,
    checkpoints: NDArray,
    margin_dist_error: float,
    method: str = "close",
    cond: str = "any",
) -> NDArray:
    """Find indices of points in `points_to_check` that are within or beyond a specified distance from any point in `checkpoints`.

    Args:
        points_to_check (NDArray): An array of points to check, of shape (N, D).
        checkpoints (NDArray): An array of reference points, of shape (M, D).
        margin_dist_error (float): The distance threshold for comparison.
        method (str, optional): Determines the comparison method.
            "close" returns indices of points within the threshold distance.
            "far" returns indices of points beyond the threshold distance.
            Defaults to "close".
        cond (str, optional): if cond='any' returns the indices of points that satisfy
            at least one condition, if cond='all' returns the indices of points that
            satisfy all the conditions. Defaults to "any".

    Returns:
        NDArray: Indices of `points_to_check` that satisfy the distance condition with
            respect to cond condition point in `checkpoints`.
    """
    valid_methods = ["close", "far"]
    if method not in valid_methods:
        raise ValueError(f"Invalid method, must be in {valid_methods}")

    valid_cond = ["any", "all"]
    if cond not in valid_cond:
        raise ValueError(f"Invalid cond, must be in {valid_cond}")

    # Compute pairwise distances: shape (N, M) where
    # N = len(points_to_check), M = len(checkpoints)
    # points_to_check[:, None, :] has shape (N, 1, 2)
    # checkpoints[None, :, :] has shape (1, M, 2)
    distances = np.linalg.norm(
        points_to_check[:, None, :] - checkpoints[None, :, :], axis=2
    )

    # Apply threshold depending on method
    if method == "close":
        mask = distances < margin_dist_error
    else:  # method == "far"
        mask = distances > margin_dist_error

    # Find which rows (i.e., points_to_check indices) have at least one True
    if cond == "any":
        valid_rows = np.any(mask, axis=1)
    else:
        valid_rows = np.all(mask, axis=1)

    # Get indices of valid rows
    return np.nonzero(valid_rows)[0].astype(int)
