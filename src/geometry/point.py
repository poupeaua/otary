"""
Point class useful to describe any kind of points
"""

import numpy as np
from shapely import Point as SPoint

from src.geometry.entity import GeometryEntity


class Point(GeometryEntity):
    """Point class"""

    def __init__(self, point: np.ndarray) -> None:
        point = np.asarray(point)
        if point.shape == (2,):
            point = point.reshape((1, 2))
        assert len(point) == 1
        super().__init__(points=point)

    @property
    def asarray(self):
        return self.points[0]

    @property
    def centroid(self):
        return self.asarray

    @property
    def shapely(self) -> SPoint:
        """Returns the Shapely.Point representation of the contour.
        See https://shapely.readthedocs.io/en/stable/reference/shapely.Point.html

        Returns:
            Point: shapely.Point object
        """
        return SPoint(self.asarray)
