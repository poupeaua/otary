"""
This file helps with geometry elements
"""

import numpy as np
import itertools

DEFAULT_MARGIN_ANGLE_ERROR = np.pi / 50

def compute_slope_angle(line: np.array) -> float:
    """Calculate the slope angle of a single line in the cartesian space

    Args:
        line (np.array): segment of shape (2, 2)

    Returns:
        float: slope angle in ]-pi/2, pi/2[
    """
    p1, p2 = line[0], line[1]
    try:
        return np.arctan((p2[1] - p1[1]) / (p2[0] - p1[0] + 1e-6))
    except ZeroDivisionError:
        if p2[1] - p1[1] > 0:
            return np.pi / 2
        else:
            return -np.pi / 2

def are_parallel(
        line1: np.array, 
        line2: np.array, 
        margin_error_angle: float=DEFAULT_MARGIN_ANGLE_ERROR,
        is_display_error_angle_enabled: bool=False
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
    if is_display_error_angle_enabled:
        print("Angle difference:", angle_difference, "Margin:", margin_error_angle)
    if angle_difference < margin_error_angle or np.abs(angle_difference - np.pi) < margin_error_angle:
        return True
    else:
        return False

def are_points_collinear(
        p1: np.array, 
        p2: np.array, 
        p3: np.array, 
        margin_error_angle: float=DEFAULT_MARGIN_ANGLE_ERROR,
        is_display_error_angle_enabled: bool=False
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
        margin_error_angle=margin_error_angle, 
        is_display_error_angle_enabled=is_display_error_angle_enabled
    )
    
def are_lines_collinear(
        line1: np.array,
        line2: np.array,
        margin_error_angle: float=DEFAULT_MARGIN_ANGLE_ERROR
    ) -> bool:
    """Verify whether two lines on the plane are collinear or not.
    This means that they are parallel and have at least three points in common.
    We needed to make all the comibnation verification in order to proove cause we could end up 
        with two points very very close by and it would end up not providing the expected result; 
        consider the following example:
    
    line1 = array([[339, 615], [564, 650]], dtype=int32)
    line2 = array([[340, 614], [611, 657]], dtype=int32)
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
    #print(are_parallel(line1=line1, line2=line2, margin_error_angle=margin_error_angle), val_list)
    if are_parallel(line1=line1, line2=line2, margin_error_angle=margin_error_angle) \
            and 1 in val_arr:
        return True
    else:
        return False