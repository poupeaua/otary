"""
Point class useful to describe any kind of points
"""

import numpy as np
from numpy.typing import NDArray
from shapely import Point as SPoint

from otary.geometry.discrete.entity import DiscreteGeometryEntity


class Point(DiscreteGeometryEntity):
    """Point class"""

    def __init__(self, point: NDArray, is_cast_int: bool = False) -> None:
        point = np.asarray(point)
        if point.shape == (2,):
            point = point.reshape((1, 2))
        assert len(point) == 1
        super().__init__(points=point, is_cast_int=is_cast_int)

    @property
    def asarray(self):
        return self.points[0]

    @asarray.setter
    def asarray(self, value: NDArray):
        """Setter for the asarray property

        Args:
            value (NDArray): value of the asarray to be changed
        """
        point = np.asarray(value)
        if point.shape == (2,):
            point = point.reshape((1, 2))
        assert len(point) == 1
        self.points = value

    @property
    def centroid(self):
        return self.asarray

    @property
    def shapely_edges(self) -> SPoint:
        """Returns the Shapely.Point representation of the point.
        See https://shapely.readthedocs.io/en/stable/reference/shapely.Point.html

        Returns:
            Point: shapely.Point object
        """
        return SPoint(self.asarray)

    @property
    def shapely_surface(self) -> SPoint:
        """Same as shapely curve in this case

        Returns:
            SPoint: shapely.Point object
        """
        return self.shapely_surface

    @property
    def area(self) -> float:
        """Compute the area of the geometry entity

        Returns:
            float: area value
        """
        return 0

    @property
    def perimeter(self) -> float:
        """Compute the perimeter of the geometry entity

        Returns:
            float: perimeter value
        """
        return 0

    @property
    def edges(self) -> NDArray:
        """Get the edges of the point which returns None since a point has no edges

        Returns:
            NDArray: empty array of shape (0, 2, 2)
        """
        return np.empty(shape=(0, 2, 2))

    @staticmethod
    def order_idxs_points_by_dist(points: NDArray, desc: bool = False) -> NDArray:
        """Beware the method expects points to be collinear.

        Given four points [p0, p1, p2, p3], we wish to have the order in which each
        point is separated.
        The one closest to the origin is placed at the origin and relative to this
        point we are able to know at which position are the other points.

        If p0 is closest to the origin and the closest points from p0 are in order
        p2, p1 and p3. Thus the array returned by the function is [0, 2, 1, 3].

        Args:
            points (NDArray): numpy array of shape (n, 2)
            desc (bool): if True returns the indices based on distances descending
                order. Otherwise ascending order which is the default.

        Returns:
            NDArray: indices of the points
        """
        distances = np.linalg.norm(x=points, axis=1)
        idxs_order_by_dist = np.argsort(distances)
        if not desc:  # change the order if in descending order
            idxs_order_by_dist = idxs_order_by_dist[::-1]
        return idxs_order_by_dist

    def distances_vertices_to_point(self, point: NDArray) -> NDArray:
        """Compute the distances to a given point

        Args:
            point (NDArray): point to which we want to compute the distances

        Returns:
            NDArray: distance to the given point
        """
        return np.linalg.norm(self.points - point, axis=1)
