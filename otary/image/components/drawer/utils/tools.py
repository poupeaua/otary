"""
Image tools module.
It contains all the utility common functions used by the Image class
"""

from typing import Any, Sequence

import numpy as np

import otary.geometry as geo


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


def is_list_elements_type(_list: Sequence | np.ndarray, _type: Any) -> bool:
    """Assert that a given list is only constituted by elements of the given type

    Args:
        _list (list): input list
        type (Any): expected type for all elements

    Returns:
        bool: True if all the element in the list are made of element of type "type"
    """
    return bool(np.all([isinstance(_list[i], _type) for i in range(len(_list))]))


def cast_geometry_to_array(objects: Sequence | np.ndarray, _type: Any) -> list:
    """Convert a list of geometric objects to array for drawing

    Warning: the limit of int range is int16 which means that the maximum value
    is 32767. If the value is higher, it will be casted to int16 and the value
    will be lost. We should not expect any X or Y coordinate to be higher than 32767.

    Args:
        objects (list): list of geometric objects
        _type (Any): type to transform into array
    """
    if _type in [geo.Point, geo.Segment, geo.Vector, geo.Polygon, geo.LinearSpline]:
        objects = [np.round(s.asarray).astype(np.int32) for s in objects]
    else:
        raise RuntimeError(f"The type {_type} is unexpected.")
    return objects


def prep_obj_draw(
    objects: Sequence | list | np.ndarray, _type: Any
) -> list | np.ndarray:
    """Preparation function for the objects to be drawn

    Args:
        objects (list | np.ndarray): list of elements to be drawn
        _type (Any): geometric type possibly of elements to be drawn

    Returns:
        np.ndarray: numpy array type
    """
    if is_list_elements_type(_list=objects, _type=_type):
        objects = cast_geometry_to_array(objects=objects, _type=_type)
    elif _type in [geo.Point]:
        try:
            # useful to let the drawing function to accept numpy array
            objects = np.asanyarray(objects).astype(np.int32)
            if len(objects.shape) != 2 or objects.shape[1] != 2:
                raise RuntimeError
        except Exception as e:
            raise RuntimeError(
                "Could not transform the input into a drawing format"
            ) from e
    elif _type in [geo.Segment, geo.Vector]:
        try:
            # useful to let the drawing function to accept numpy array
            objects = np.asanyarray(objects).astype(np.int32)
            if (
                len(objects.shape) != 3
                or objects.shape[1] != 2
                or objects.shape[2] != 2
            ):
                raise RuntimeError
        except Exception as e:
            raise RuntimeError(
                "Could not transform the input into a drawing format"
            ) from e
    else:
        raise RuntimeError(
            f"Unexpected type for the objects to be drawn. Expected {_type}."
        )
    return objects
