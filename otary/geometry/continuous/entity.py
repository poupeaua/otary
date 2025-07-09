"""
ContinuousGeometryEntity module class
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np

from otary.geometry.entity import GeometryEntity
from otary.geometry import Polygon, Rectangle


class ContinuousGeometryEntity(GeometryEntity, ABC):
    """
    ContinuousGeometryEntity class which is the abstract base class for
    continuous or smooth geometry objects like circles, ellipse, etc...
    """

    DEFAULT_N_POINTS_POLYGONAL_APPROX = 1000

    def __init__(
        self, n_points_polygonal_approx: int = DEFAULT_N_POINTS_POLYGONAL_APPROX
    ):
        """Initialize a ContinuousGeometryEntity object

        Args:
            n_points_polygonal_approx (int, optional): n points to be used in
                the polygonal approximation.
                Defaults to DEFAULT_N_POINTS_POLYGONAL_APPROX.
        """
        self._n_points_polygonal_approx = n_points_polygonal_approx

    # --------------------------------- PROPERTIES ------------------------------------

    @property
    def n_points_polygonal_approx(self) -> int:
        """Get the number of points for the polygonal approximation.

        Returns:
            int: The number of points used in the polygonal approximation.
        """
        return self._n_points_polygonal_approx

    @n_points_polygonal_approx.setter
    def n_points_polygonal_approx(self, value):
        """
        Set the number of points for the polygonal approximation.
        This would also update the polygonal approximation of the geometry entity.

        Args:
            value (int): The number of points to be used in the polygonal approximation.
        """
        self._n_points_polygonal_approx = value
        self.update_polyapprox()

    @property
    def polyaprox(self) -> Polygon:
        """Generate a polygonal approximation of the continuous geometry entity
        No setter is defined for this property as it is a read-only property.

        Returns:
            Polygon: polygonal approximation of the continuous geometry entity
        """
        return self._polyapprox

    @abstractmethod
    def polygonal_approx(self, n_points: int, is_cast_int: bool) -> Polygon:
        """Generate a polygonal approximation of the continuous geometry entity

        Args:
            n_points (int): number of points that make up the polygonal
                approximation. The bigger the better to obtain more precise
                results in intersection or other similar computations.

        Returns:
            Polygon: polygonal approximation of the continuous geometry entity
        """

    @property
    def xmax(self) -> float:
        """Get the maximum X coordinate of the geometry entity

        Returns:
            np.ndarray: 2D point
        """
        return self.polyaprox.xmax

    @property
    def xmin(self) -> float:
        """Get the minimum X coordinate of the geometry entity

        Returns:
            np.ndarray: 2D point
        """
        return self.polyaprox.xmin

    @property
    def ymax(self) -> float:
        """Get the maximum Y coordinate of the geometry entity

        Returns:
            np.ndarray: 2D point
        """
        return self.polyaprox.ymax

    @property
    def ymin(self) -> float:
        """Get the minimum Y coordinate of the geometry entity

        Returns:
            np.ndarray: 2D point
        """
        return self.polyaprox.ymin

    # ---------------------------- MODIFICATION METHODS -------------------------------

    # done in the derived classes

    # ------------------------------- CLASSIC METHODS ---------------------------------

    def update_polyapprox(self) -> None:
        """Update the polygonal approximation of the continuous geometry entity"""
        self._polyapprox = self.polygonal_approx(
            n_points=self.n_points_polygonal_approx, is_cast_int=False
        )

    def enclosing_axis_aligned_bbox(self) -> Rectangle:
        """Compute the smallest area enclosing Axis-Aligned Bounding Box (AABB)
        See: https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html

        Returns:
            Rectangle: Rectangle object
        """
        topleft_x, topleft_y, width, height = cv2.boundingRect(
            array=self.polyaprox.asarray
        )
        bbox = np.array(
            [
                [topleft_x, topleft_y],
                [topleft_x + width, topleft_y],
                [topleft_x + width, topleft_y + height],
                [topleft_x, topleft_y + height],
            ]
        )
        return Rectangle(bbox)

    def enclosing_oriented_bbox(self) -> Rectangle:
        """Compute the smallest area enclosing Oriented Bounding Box (OBB)
        See: https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html

        Returns:
            Rectangle: Rectangle object
        """
        rect = cv2.minAreaRect(self.polyaprox.asarray)
        bbox = cv2.boxPoints(rect)
        return Rectangle(bbox)

    def enclosing_convex_hull(self) -> "Polygon":
        """Compute the smallest area enclosing Convex Hull
        See: https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html

        Returns:
            Polygon: Polygon object
        """
        # pylint: disable=import-outside-toplevel
        from otary.geometry import Polygon

        convexhull = np.squeeze(cv2.convexHull(self.polyaprox.asarray))
        return Polygon(convexhull)
