"""
Point class useful to describe any kind of points
"""

import numpy as np
from shapely import Point as SPoint

from src.geometry.entity import GeometryEntity


class Point(GeometryEntity):
    """Point class"""

    def __init__(self, point: np.ndarray, is_cast_int: bool = False) -> None:
        point = np.asarray(point)
        if point.shape == (2,):
            point = point.reshape((1, 2))
        assert len(point) == 1
        super().__init__(points=point, is_cast_int=is_cast_int)

    @property
    def asarray(self):
        return self.points[0]

    @property
    def centroid(self):
        return self.asarray

    @property
    def shapely_curve(self) -> SPoint:
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

    @staticmethod
    def order_idxs_points_by_dist(points: np.ndarray, desc: bool = False) -> np.ndarray:
        """Beware the method expects points to be collinear.

        Given four points [p0, p1, p2, p3], we wish to have the order in which each
        point is separated.
        The one closest to the origin is placed at the origin and relative to this
        point we are able to know at which position are the other points.

        If p0 is closest to the origin and the closest points from p0 are in order
        p2, p1 and p3. Thus the array returned by the function is [0, 2, 1, 3].

        Args:
            points (np.ndarray): numpy array of shape (n, 2)
            desc (bool): if True returns the indices based on distances descending
                order. Otherwise ascending order which is the default.

        Returns:
            np.ndarray: indices of the points
        """
        distances = np.linalg.norm(x=points, axis=1)
        idxs_order_by_dist = np.argsort(distances)
        if not desc:  # change the order if in descending order
            idxs_order_by_dist = idxs_order_by_dist[::-1]
        return idxs_order_by_dist
