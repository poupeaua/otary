"""
Module to facilitate imports in geometry
"""

__all__ = [
    "DEFAULT_MARGIN_ANGLE_ERROR",
    "Point",
    "Segment",
    "Vector",
    "LinearSpline",
    "VectorizedLinearSpline",
    "Polygon",
    "Triangle",
    "Rectangle",
    "Ellipse",
    "Circle",
]

# pylint: disable=cyclic-import
from otary.geometry.utils.constants import DEFAULT_MARGIN_ANGLE_ERROR
from otary.geometry.discrete.point import Point
from otary.geometry.discrete.linear.segment import Segment
from otary.geometry.discrete.linear.directed.vector import Vector
from otary.geometry.discrete.linear.spline import LinearSpline
from otary.geometry.discrete.linear.directed.vectorized_linear_spline import (
    VectorizedLinearSpline,
)
from otary.geometry.discrete.shape.polygon import Polygon
from otary.geometry.discrete.shape.triangle import Triangle
from otary.geometry.discrete.shape.rectangle import Rectangle
from otary.geometry.continuous.shape.ellipse import Ellipse
from otary.geometry.continuous.shape.circle import Circle
