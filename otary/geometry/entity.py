"""
GeometryEntity module which allows to define transformation and property shared
by all type of geometry objects
"""

from __future__ import annotations

from typing import Optional, Self, TYPE_CHECKING
from abc import ABC, abstractmethod
from shapely import (
    GeometryCollection,
    MultiPoint,
    Point as SPoint,
    LineString,
    MultiLineString,
)

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from otary.geometry import Polygon, Rectangle  # pragma: no cover


class GeometryEntity(ABC):
    """GeometryEntity class which is the abstract base class for all geometry classes"""

    # --------------------------------- PROPERTIES ------------------------------------

    @property
    @abstractmethod
    def shapely_edges(self) -> GeometryCollection:
        """Representation of the geometric object in the shapely library
        as a geometrical object defined only as a curve with no area. Particularly
        useful to look for points intersections
        """

    @property
    @abstractmethod
    def shapely_surface(self) -> GeometryCollection:
        """Representation of the geometric object in the shapely library
        as a geometrical object with an area and a border. Particularly useful
        to check if two geometrical objects are contained within each other or not.
        """

    @property
    @abstractmethod
    def area(self) -> float:
        """Compute the area of the geometry entity

        Returns:
            float: area value
        """

    @property
    @abstractmethod
    def perimeter(self) -> float:
        """Compute the perimeter of the geometry entity

        Returns:
            float: perimeter value
        """

    @property
    @abstractmethod
    def centroid(self) -> NDArray:
        """Compute the centroid point which can be seen as the center of gravity of
        the shape

        Returns:
            NDArray: centroid point
        """

    @property
    @abstractmethod
    def xmax(self) -> float:
        """Get the maximum X coordinate of the geometry entity

        Returns:
            NDArray: 2D point
        """

    @property
    @abstractmethod
    def xmin(self) -> float:
        """Get the minimum X coordinate of the geometry entity

        Returns:
            NDArray: 2D point
        """

    @property
    @abstractmethod
    def ymax(self) -> float:
        """Get the maximum Y coordinate of the geometry entity

        Returns:
            NDArray: 2D point
        """

    @property
    @abstractmethod
    def ymin(self) -> float:
        """Get the minimum Y coordinate of the geometry entity

        Returns:
            NDArray: 2D point
        """

    # ---------------------------- MODIFICATION METHODS -------------------------------

    @abstractmethod
    def rotate(
        self,
        angle: float,
        is_degree: bool = False,
        is_clockwise: bool = True,
        pivot: Optional[NDArray] = None,
    ) -> Self:
        """Rotate the geometry entity object.
        A pivot point can be passed as an argument to rotate the object around the pivot

        Args:
            angle (float): rotation angle
            is_degree (bool, optional): whether the angle is in degree or radian.
                Defaults to False which means radians.
            is_clockwise (bool, optional): whether the rotation is clockwise or
                counter-clockwise. Defaults to True.
            pivot (NDArray, optional): pivot point.
                Defaults to None which means that by default the centroid point of
                the shape is taken as the pivot point.

        Returns:
            GeometryEntity: rotated geometry entity object.
        """

    @abstractmethod
    def shift(self, vector: NDArray) -> Self:
        """Shift the geometry entity by the vector direction

        Args:
            vector (NDArray): vector that describes the shift as a array with
                two elements. Example: [2, -8] which describes the
                vector [[0, 0], [2, -8]]. The vector can also be a vector of shape
                (2, 2) of the form [[2, 6], [1, 3]].

        Returns:
            GeometryEntity: shifted geometrical object
        """

    @abstractmethod
    def normalize(self, x: float, y: float) -> Self:
        """Normalize the geometry entity by dividing the points by a norm on the
        x and y coordinates.

        Args:
            x (float): x coordinate norm
            y (float): y coordinate norm

        Returns:
            GeometryEntity: normalized GeometryEntity
        """

    # ------------------------------- CLASSIC METHODS ---------------------------------

    @abstractmethod
    def copy(self) -> Self:
        """Create a copy of the geometry entity object

        Returns:
            GeometryEntity: copy of the geometry entity object
        """

    def intersection(self, other: GeometryEntity, only_points: bool = True) -> NDArray:
        """Compute the intersections between two geometric objects.
        If the only_points parameter is True, then we only consider intersection points
        as valid. We can not have another type of intersection.

        Args:
            other (GeometryEntity): other GeometryEntity object
            only_points (bool, optional): whether to consider only points.
                Defaults to True.

        Returns:
            NDArray: list of n points of shape (n, 2)
        """
        it = self.shapely_edges.intersection(other=other.shapely_edges)

        if isinstance(it, SPoint):  # only one intersection point
            return np.array([[it.x, it.y]])
        if isinstance(it, MultiPoint):  # several intersection points
            return np.asanyarray([[pt.x, pt.y] for pt in it.geoms])
        if isinstance(it, LineString) and not only_points:  # one intersection line
            return NotImplemented
        if isinstance(it, MultiLineString) and not only_points:  # multilines
            return NotImplemented
        if isinstance(it, GeometryCollection):  # lines and pts
            return NotImplemented

        return np.array([])

    @abstractmethod
    def enclosing_axis_aligned_bbox(self) -> Rectangle:
        """Compute the smallest area enclosing Axis-Aligned Bounding Box (AABB)
        See: https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html

        Returns:
            Rectangle: Rectangle object
        """

    @abstractmethod
    def enclosing_oriented_bbox(self) -> Rectangle:
        """Compute the smallest area enclosing Oriented Bounding Box (OBB)
        See: https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html

        Returns:
            Rectangle: Rectangle object
        """

    @abstractmethod
    def enclosing_convex_hull(self) -> Polygon:
        """Compute the smallest area enclosing Convex Hull
        See: https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html

        Returns:
            Polygon: Polygon object
        """
