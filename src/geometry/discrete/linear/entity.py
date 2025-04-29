"""
LinearEntity class useful to describe any kind of linear object
"""

from abc import ABC

import numpy as np

from shapely import LineString

from src.geometry.discrete.entity import DiscreteGeometryEntity


class LinearEntity(DiscreteGeometryEntity, ABC):
    """Define Linear objects"""

    @property
    def length(self) -> float:
        """Compute the length of the linear object

        Returns:
            float: length of the curve
        """
        _length: float = 0
        for pt1, pt2 in zip(self.asarray[:-1], self.asarray[1:]):
            _length += float(np.linalg.norm(pt1 - pt2))
        return _length

    @property
    def perimeter(self) -> float:
        """Perimeter of the segment which we define to be its length

        Returns:
            float: segment perimeter
        """
        return self.length

    @property
    def shapely_edges(self) -> LineString:
        """Returns the Shapely.LineString representation of the segment.
        See https://shapely.readthedocs.io/en/stable/reference/shapely.LineString.html

        Returns:
            LineString: shapely.LineString object
        """
        return LineString(coordinates=self.asarray)

    @property
    def shapely_surface(self) -> LineString:
        """Returns the Shapely.LineString representation of the segment.
        See https://shapely.readthedocs.io/en/stable/reference/shapely.LineString.html

        Returns:
            LineString: shapely.LineString object
        """
        return self.shapely_edges

    @property
    def edges(self) -> list:
        """Get the edges of the linear spline

        Returns:
            list: edges of the linear spline
        """
        return np.stack([self.points, np.roll(self.points, shift=-1, axis=0)], axis=1)[
            :-1, :, :
        ]
