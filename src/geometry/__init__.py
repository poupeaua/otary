"""
Module to facilitate imports in geometry
"""

# ruff: noqa: F401
# pylint: disable=cyclic-import
from src.geometry.constants import DEFAULT_MARGIN_ANGLE_ERROR
from src.geometry.entity import GeometryEntity
from src.geometry.point import Point
from src.geometry.segment import Segment
from src.geometry.vector import Vector
from src.geometry.polygon import Polygon
from src.geometry.triangle import Triangle
from src.geometry.rectangle import Rectangle
from src.geometry.linear_spline import LinearSpline
from src.geometry.vectorized_linear_spline import VectorizedLinearSpline
