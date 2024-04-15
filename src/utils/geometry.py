"""
This file helps with geometry elements
"""

import numpy as np
import itertools
import logging

DEFAULT_MARGIN_ANGLE_ERROR = np.pi / 50

def rotate_bbox_around_img_center(img: np.ndarray, bbox: np.ndarray):
    center_img_vector = (np.array([img.shape[1], img.shape[0]]) / 2).astype(int)
    pmin = np.array([np.min(bbox[:, 0]), np.min(bbox[:, 1])])
    pmax = np.array([np.max(bbox[:, 0]), np.max(bbox[:, 1])])
    height_rect = pmax[1] - pmin[1]
    width_rect = pmax[0] - pmin[0]
    vect_shift = center_img_vector - pmin
    new_opposite_point = pmin + 2 * vect_shift
    [[2, 0], [81, 0], [81, 28], [2, 28]]
    new_bbox = np.array([
        new_opposite_point - np.array([width_rect, height_rect]),
        new_opposite_point - np.array([0, height_rect]),
        new_opposite_point,
        new_opposite_point - np.array([width_rect, 0])
    ]).tolist()
    return new_bbox
    

def slope(line: np.ndarray) -> float:
    p1, p2 = line[0], line[1]
    try:
        return (p2[1] - p1[1]) / (p2[0] - p1[0])
    except ZeroDivisionError:
        return np.inf
    
def intercept(line: np.ndarray) -> float:
    p1 = line[0]
    slope = slope(line)
    try:
        return p1[1] - slope * p1[0]
    except Exception:
        return None

def compute_slope_angle(line: np.ndarray, degree: bool=False) -> float:
    """Calculate the slope angle of a single line in the cartesian space

    Args:
        line (np.array): segment of shape (2, 2)
        degree (bool): whether to output the result in degree. By default in radian.

    Returns:
        float: slope angle in ]-pi/2, pi/2[
    """
    p1, p2 = line[0], line[1]
    try:
        angle = np.arctan((p2[1] - p1[1]) / (p2[0] - p1[0] + 1e-9))
    except ZeroDivisionError:
        angle = np.pi / 2
    if degree:
        angle = np.rad2deg(angle)
    return angle

def are_parallel(
        line1: np.ndarray, 
        line2: np.ndarray, 
        margin_error_angle: float=DEFAULT_MARGIN_ANGLE_ERROR
    ) -> bool:
    """Check if two lines are parallel by calculating the slope of the two lines
    
    Angle Difference = |theta_0 - theta_1| mod pi
    Beware this will always return positive results due to the modulo.
    So we took into account the special case where angle difference = np.pi - epsilon ~ 3.139,
    this implies also two parralel lines.

    Args:
        line1 (np.array): line of shape (2, 2)
        line2 (np.array): line of shape (2, 2)
        margin_error_angle (float, optional): Threshold value for validating if the lines are parallel. Defaults to DEFAULT_MARGIN_ANGLE_ERROR.

    Returns:
        bool: whether we qualify the lines as parallel or not
    """
    angle1 = compute_slope_angle(line=line1)
    angle2 = compute_slope_angle(line=line2)
    
    angle_difference = np.mod(np.abs(angle1 - angle2), np.pi) 
    logging.debug("Angle difference:", angle_difference, "Margin:", margin_error_angle)
    if angle_difference < margin_error_angle or np.abs(angle_difference - np.pi) < margin_error_angle:
        return True
    else:
        return False

def are_points_collinear(
        p1: np.ndarray, 
        p2: np.ndarray, 
        p3: np.ndarray, 
        margin_error_angle: float=DEFAULT_MARGIN_ANGLE_ERROR
    ) -> bool:
    """Verify whether three points on the plane are collinear or not.
    Method by angle or slope: For three points, slope of any pair of points must be same as other pair.

    Args:
        p1 (np.array): point of shape (2,)
        p2 (np.array): point of shape (2,)
        p3 (np.array): point of shape (2,)
        margin_error_angle (float, optional): Threshold value for validating collinearity. 
            Defaults to DEFAULT_MARGIN_ANGLE_ERROR.

    Returns:
        bool: 1 if colinear, 0 otherwise
    """
    p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
    
    # 2 or 3 points equal
    if not np.logical_or(*(p1 - p2)) or not np.logical_or(*(p1 - p3)) \
            or not np.logical_or(*(p2 - p3)):
        return True
    
    line1, line2 = np.array([p1, p2]), np.array([p1, p3])
    return are_parallel(
        line1=line1, 
        line2=line2, 
        margin_error_angle=margin_error_angle
    )
    
def are_lines_collinear(
        line1: np.ndarray,
        line2: np.ndarray,
        margin_error_angle: float=DEFAULT_MARGIN_ANGLE_ERROR
    ) -> bool:
    """Verify whether two lines on the plane are collinear or not.
    This means that they are parallel and have at least three points in common.
    We needed to make all the comibnation verification in order to proove cause we could end up 
        with two points very very close by and it would end up not providing the expected result; 
        consider the following example:
    
    line1 = array([[339, 615], [564, 650]], dtype=int32)
    line2 = array([[340, 614], [611, 657]], dtype=int32)
    are_lines_collinear(line1, line2)
    Angle difference: 0.9397169393235674 Margin: 0.06283185307179587
    False
    
    Only because [339, 615] and [340, 614] are really close and do not provide the 
    appropriate slope.

    Args:
        line1 (np.array): line of shape (2, 2)
        line2 (np.array): line of shape (2, 2)
        margin_error_angle (float, optional): Threshold value for validating collinearity. 
            Defaults to np.pi/150.

    Returns:
        bool: 1 if colinear, 0 otherwise
    """
    cur2lines = np.array([line1, line2])
    points = np.concatenate(cur2lines, axis=0)
    val_arr = np.zeros(shape=4)
    for i, combi in enumerate(itertools.combinations(np.linspace(0, 3, 4, dtype=int), 3)):
        val_arr[i] = are_points_collinear(
            p1=points[combi[0]], 
            p2=points[combi[1]], 
            p3=points[combi[2]], 
            margin_error_angle=margin_error_angle
        )
    logging.debug(f"{are_parallel(line1, line2, margin_error_angle=margin_error_angle)}{val_arr}")
    if are_parallel(line1=line1, line2=line2, margin_error_angle=margin_error_angle) \
            and 1 in val_arr:
        return True
    else:
        return False
    
def assert_is_array_of_lines(lines: np.ndarray):
    """Check whether the array has the appropriate shape

    Args:
        lines (np.array): array of expected shape (n, 2, 2)


    Returns:
        bool: True if array of lines else raise error
    """
    if lines.shape[1:] != (2, 2):
        raise RuntimeError(
            f"The input array has not the expected array of lines shape (n, 2, 2).\
            It has the following shape {lines.shape}."
        )
    else:
        return True