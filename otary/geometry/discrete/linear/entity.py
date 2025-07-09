"""
LinearEntity class useful to describe any kind of linear object
"""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from shapely import LineString

from otary.geometry.discrete.entity import DiscreteGeometryEntity


class LinearEntity(DiscreteGeometryEntity, ABC):
    """Define Linear objects"""

    @property
    def length(self) -> float:
        """Compute the length of the linear object.

        Returns:
            float: length of the curve
        """
        return np.sum(self.lengths)

    @property
    def perimeter(self) -> float:
        """Perimeter of the segment which we define to be its length

        Returns:
            float: segment perimeter
        """
        return self.length

    @property
    def area(self) -> float:
        """Area of the segment which we define to be its length

        Returns:
            float: segment area
        """
        return 0

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
    def edges(self) -> NDArray:
        """Get the edges of the linear spline

        Returns:
            NDArray: edges of the linear spline
        """
        return np.stack([self.points, np.roll(self.points, shift=-1, axis=0)], axis=1)[
            :-1, :, :
        ]

    @property
    @abstractmethod
    def midpoint(self) -> NDArray:
        """Returns the mid-point of the linear entity that is within or
        along the entity.

        This method can be useful in the case of a curved linear entity. The
        centroid is not necessarily along the curved linear entity, the mid-point is.

        Returns:
            NDArray: 2D point
        """

    def __str__(self) -> str:
        return (
            self.__class__.__name__
            + "(start="
            + self.asarray[0].tolist().__str__()
            + ", end="
            + self.asarray[-1].tolist().__str__()
            + ")"
        )

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + "(start="
            + self.asarray[0].tolist().__str__()
            + ", end="
            + self.asarray[-1].tolist().__str__()
            + ")"
        )
