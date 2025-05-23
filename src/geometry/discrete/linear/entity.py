"""
LinearEntity class useful to describe any kind of linear object
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from abc import ABC

import numpy as np

from shapely import LineString

from src.geometry.discrete.entity import DiscreteGeometryEntity

if TYPE_CHECKING:
    from src.geometry.discrete.shape.polygon import Polygon


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
    def edges(self) -> np.ndarray:
        """Get the edges of the linear spline

        Returns:
            np.ndarray: edges of the linear spline
        """
        return np.stack([self.points, np.roll(self.points, shift=-1, axis=0)], axis=1)[
            :-1, :, :
        ]

    @staticmethod
    def linear_entities_to_polygon(
        linear_entities: list[LinearEntity], connected: bool = False
    ) -> Polygon:
        """Convert a list of linear entities to polygon.

        Returns:
            Polygon: polygon representation of the linear entity
        """
        from src.geometry.discrete.shape.polygon import Polygon

        points = []
        for linear_entity in linear_entities:
            if not isinstance(linear_entity, LinearEntity):
                raise TypeError(
                    f"Expected a list of LinearEntity, but got {type(linear_entity)}"
                )
            if connected:
                # if we assume all linear entites sorted and connected
                # we need to remove the last point of each linear entity
                points.append(linear_entity.points[:-1, :])
            else:
                points.append(linear_entity.points)
        points = np.concatenate(points, axis=0)
        return Polygon(points=points)

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
