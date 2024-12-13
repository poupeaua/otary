"""
Image tools module.
It contains all the utility common functions used by the Image class
"""

from typing import Any
import numpy as np

import src.geometry as geo


def is_color_tuple(color: Any) -> bool:
    """Identify if the input color parameter is in the expected format for a color

    Args:
        color (tuple): an expected python object to define a color

    Returns:
        bool: True if the input is a good color, False otherwise
    """
    cond = bool(
        isinstance(color, tuple)
        and len(color) == 3
        and np.all([isinstance(c, int) and 0 <= c <= 255 for c in color])
    )
    return cond


def is_list_elements_type(_list: list | np.ndarray, _type: Any) -> bool:
    """Assert that a given list is only constituted by elements of the given type

    Args:
        _list (list): input list
        type (Any): expected type for all elements

    Returns:
        bool: True if all the element in the list are made of element of type "type"
    """
    return bool(np.all([isinstance(_list[i], _type) for i in range(len(_list))]))


def cast_geometry_to_array(objects: list | np.ndarray, _type: Any):
    """Convert a list of geometric objects to array for drawing

    Args:
        objects (list): list of geometric objects
        _type (Any): type to transform into array
    """
    if _type in [geo.Point, geo.Segment, geo.Vector]:
        objects = [s.asarray.astype(np.int64) for s in objects]
    elif _type == geo.Polygon:
        objects = [s.segments.astype(np.int64) for s in objects]
    else:
        raise RuntimeError(f"The type {_type} is unexpected.")
    return objects


def prep_obj_draw(objects: list | np.ndarray, _type: Any) -> np.ndarray:
    """Preparation function for the objects to be drawn

    Args:
        objects (list | np.ndarray): list of elements to be drawn
        _type (Any): geometric type possibly of elements to be drawn

    Returns:
        np.ndarray: numpy array type
    """
    if is_list_elements_type(_list=objects, _type=_type):
        objects = cast_geometry_to_array(objects=objects, _type=_type)
    try:
        objects = np.asanyarray(objects).astype(np.int64)
    except Exception as e:
        raise RuntimeError("Could not prepare the objects to draw") from e
    return objects


def interpolate_color(alpha: float) -> tuple:
    """Interpolates between red, yellow, and green based on the parameter alpha.

    Args:
        alpha (float): Parameter ranging from 0 to 1.
            0 corresponds to red, 0.5 to yellow, and 1 to green.

    Returns:
        tuple: RGB color as a tuple (R, G, B) where each value is in the range [0, 255].
    """
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")

    # Define RGB colors
    red = (255, 0, 0)
    yellow = (255, 255, 0)
    green = (0, 255, 0)

    if alpha <= 0.5:
        # Interpolate between red and yellow
        t = alpha * 2
        r = int((1 - t) * red[0] + t * yellow[0])
        g = int((1 - t) * red[1] + t * yellow[1])
        b = int((1 - t) * red[2] + t * yellow[2])
    else:
        # Interpolate between yellow and green
        t = (alpha - 0.5) * 2
        r = int((1 - t) * yellow[0] + t * green[0])
        g = int((1 - t) * yellow[1] + t * green[1])
        b = int((1 - t) * yellow[2] + t * green[2])

    return (r, g, b)
