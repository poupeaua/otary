"""
Circle Geometric Object
"""

from typing import Self, Optional
import math

import numpy as np
from numpy.typing import NDArray

from shapely import Polygon as SPolygon, LinearRing

from otary.geometry.utils.tools import rotate_2d_points
from otary.geometry.continuous.entity import ContinuousGeometryEntity
from otary.geometry import Ellipse, Polygon
from otary.utils.tools import assert_transform_shift_vector


class Circle(Ellipse):
    """Circle geometrical object"""

    def __init__(
        self,
        center: NDArray,
        radius: float,
        n_points_polygonal_approx: int = ContinuousGeometryEntity.DEFAULT_N_POINTS_POLYGONAL_APPROX,
    ):
        """Initialize a Circle geometrical object

        Args:
            center (NDArray): center 2D point
            radius (float): radius value
            n_points_polygonal_approx (int, optional): number of points to be used in
                the polygonal approximation of the circle. Defaults to
                ContinuousGeometryEntity.DEFAULT_N_POINTS_POLYGONAL_APPROX.
        """
        super().__init__(
            foci1=center,
            foci2=center,
            semi_major_axis=radius,
            n_points_polygonal_approx=n_points_polygonal_approx,
        )
        self.center = center
        self.radius = radius
        self.update_polyapprox()

    # --------------------------------- PROPERTIES ------------------------------------

    @property
    def perimeter(self) -> float:
        """Perimeter of the circle

        Returns:
            float: perimeter value
        """
        return 2 * math.pi * self.radius

    @property
    def centroid(self) -> NDArray:
        """Center of the circle

        Returns:
            float: center 2D point
        """
        return self.center

    @property
    def shapely_surface(self) -> SPolygon:
        """Returns the Shapely.Polygon as an surface representation of the Circle.
        See https://shapely.readthedocs.io/en/stable/reference/shapely.Polygon.html

        Returns:
            Polygon: shapely.Polygon object
        """
        return SPolygon(self.polyaprox.asarray, holes=None)

    @property
    def shapely_edges(self) -> LinearRing:
        """Returns the Shapely.LinearRing as a curve representation of the Circle.
        See https://shapely.readthedocs.io/en/stable/reference/shapely.LinearRing.html

        Returns:
            LinearRing: shapely.LinearRing object
        """
        return LinearRing(coordinates=self.polyaprox.asarray)

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

    def curvature(self, point: Optional[NDArray] = None) -> float:
        """Curvature of circle is a constant and does not depend on a position of
        a point

        Returns:
            float: curvature value
        """
        return 1 / self.radius

    @property
    def xmax(self) -> float:
        """Get the maximum X coordinate of the geometry entity

        Returns:
            NDArray: 2D point
        """
        return self.center[0] + self.radius

    @property
    def xmin(self) -> float:
        """Get the minimum X coordinate of the geometry entity

        Returns:
            NDArray: 2D point
        """
        return self.center[0] - self.radius

    @property
    def ymax(self) -> float:
        """Get the maximum Y coordinate of the geometry entity

        Returns:
            NDArray: 2D point
        """
        return self.center[1] + self.radius

    @property
    def ymin(self) -> float:
        """Get the minimum Y coordinate of the geometry entity

        Returns:
            NDArray: 2D point
        """
        return self.center[1] - self.radius

    # ---------------------------- MODIFICATION METHODS -------------------------------

    def rotate(
        self,
        angle: float,
        is_degree: bool = False,
        is_clockwise: bool = True,
        pivot: Optional[NDArray] = None,
    ) -> Self:
        """Rotate the circle around a pivot point.

        Args:
            angle (float): angle by which to rotate the circle
            is_degree (bool, optional): whether the angle is in degrees. Defaults to False.
            is_clockwise (bool, optional): whether the rotation is clockwise. Defaults to True.
            pivot (Optional[NDArray], optional): pivot point around which to rotate. Defaults to None.

        Returns:
            Self: rotated circle object
        """
        if pivot is None:
            # If no pivot is given, the circle is rotated around its center
            # and thus is not modified
            return self

        self.center = rotate_2d_points(
            points=self.center,
            angle=angle,
            is_degree=is_degree,
            is_clockwise=is_clockwise,
            pivot=pivot,
        )
        self.update_polyapprox()
        return self

    def shift(self, vector: NDArray) -> Self:
        """Shift the circle by a given vector.

        Args:
            vector (NDArray): 2D vector by which to shift the circle

        Returns:
            Self: shifted circle object
        """
        vector = assert_transform_shift_vector(vector=vector)
        self.center += vector
        self.update_polyapprox()
        return self

    def normalize(self, x: float, y: float) -> Self:
        """Normalize the circle by dividing the points by a norm on the x and y
        coordinates. This does not change the circle radius.

        Args:
            x (float): x coordinate norm
            y (float): y coordinate norm

        Returns:
            Self: normalized circle object
        """
        self.center = self.center / np.array([x, y])
        self.update_polyapprox()
        return self

    # ------------------------------- CLASSIC METHODS ---------------------------------

    def copy(self) -> Self:
        """Copy the circle object

        Returns:
            Self: copied circle object
        """
        return type(self)(
            center=self.center,
            radius=self.radius,
            n_points_polygonal_approx=self.n_points_polygonal_approx,
        )

    def __str__(self) -> str:
        return f"Circle(center={self.center}, radius={self.radius})"

    def __repr__(self):
        return f"Circle(center={self.center}, radius={self.radius})"
