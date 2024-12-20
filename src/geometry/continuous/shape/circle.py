"""
Circle Geometric Object
"""

from typing import Self
import math

import numpy as np

from shapely import Polygon as SPolygon, LinearRing

from src.geometry.continuous.entity import ContinuousGeometryEntity
from src.geometry import Ellipse, Polygon


class Circle(Ellipse):
    """Circle geometrical object"""

    def __init__(
        self,
        center: np.ndarray,
        radius: float,
        n_points_polygonal_approx: int = ContinuousGeometryEntity.DEFAULT_N_POINTS_POLYGONAL_APPROX,
    ):
        super().__init__(
            foci1=center,
            foci2=center,
            semi_major_axis=radius,
            n_points_polygonal_approx=n_points_polygonal_approx,
        )
        self.center = center
        self.radius = radius

    @property
    def perimeter(self) -> float:
        return 2 * math.pi * self.radius

    @property
    def centroid(self) -> float:
        return self.center

    @property
    def shapely_surface(self) -> SPolygon:
        """Returns the Shapely.Polygon as an surface representation of the Polygon.
        See https://shapely.readthedocs.io/en/stable/reference/shapely.Polygon.html

        Returns:
            Polygon: shapely.Polygon object
        """
        return SPolygon(
            self.polygonal_approx(n_points=self.n_points_polygonal_approx), holes=None
        )

    @property
    def shapely_edges(self) -> LinearRing:
        """Returns the Shapely.LinearRing as a curve representation of the Polygon.
        See https://shapely.readthedocs.io/en/stable/reference/shapely.LinearRing.html

        Returns:
            LinearRing: shapely.LinearRing object
        """
        return LinearRing(
            coordinates=self.polygonal_approx(n_points=self.n_points_polygonal_approx)
        )

    def polygonal_approx(self, n_points: int, is_cast_int: bool = False) -> Polygon:
        """Generate a Polygon object that is an approximation of the circle
        as a discrete geometrical object made up of only points and segments.

        Args:
            n_points (int): number of points that make up the circle
                polygonal approximation
            is_cast_int (bool): whether to cast to int the points coordinates or
                not. Defaults to False

        Returns:
            Polygon: Polygon representing the circle as a succession of n points

        """
        points = []
        for theta in np.linspace(0, 2 * math.pi, n_points):
            x = self.center[0] + self.radius * math.cos(theta)
            y = self.center[1] + self.radius * math.sin(theta)
            points.append([x, y])

        poly = Polygon(points=np.asarray(points), is_cast_int=is_cast_int)
        return poly

    def curvature(self) -> float:
        """Curvature of circle is a constant and does not depend on a position of
        a point

        Returns:
            float: curvature value
        """
        return 1 / self.radius

    def copy(self) -> Self:
        return type(self)(
            center=self.center,
            radius=self.radius,
            n_points_polygonal_approx=self.n_points_polygonal_approx,
        )
