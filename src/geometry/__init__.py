"""
Module to facilitate imports in geometry
"""

# ruff: noqa: F401
# pylint: disable=cyclic-import
from src.geometry.utils.constants import DEFAULT_MARGIN_ANGLE_ERROR
from src.geometry.entity import GeometryEntity
from src.geometry.linear.entity import LinearEntity
from src.geometry.linear.directed.entity import DirectedLinearEntity
from src.geometry.point import Point
from src.geometry.linear.segment import Segment
from src.geometry.linear.directed.vector import Vector
from src.geometry.linear.spline import LinearSpline
from src.geometry.linear.directed.vectorized_linear_spline import VectorizedLinearSpline
from src.geometry.shape.polygon import Polygon
from src.geometry.shape.triangle import Triangle
from src.geometry.shape.rectangle import Rectangle
