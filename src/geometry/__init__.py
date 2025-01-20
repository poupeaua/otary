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
from src.geometry.utils.constants import DEFAULT_MARGIN_ANGLE_ERROR
from src.geometry.discrete.point import Point
from src.geometry.discrete.linear.segment import Segment
from src.geometry.discrete.linear.directed.vector import Vector
from src.geometry.discrete.linear.spline import LinearSpline
from src.geometry.discrete.linear.directed.vectorized_linear_spline import (
    VectorizedLinearSpline,
)
from src.geometry.discrete.shape.polygon import Polygon
from src.geometry.discrete.shape.triangle import Triangle
from src.geometry.discrete.shape.rectangle import Rectangle
from src.geometry.continuous.shape.ellipse import Ellipse
from src.geometry.continuous.shape.circle import Circle
