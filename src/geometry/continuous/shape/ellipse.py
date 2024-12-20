"""
Ellipse Geometric Object
"""

from typing import Optional

import math
import numpy as np

from src.geometry.continuous.entity import ContinuousGeometryEntity


class Ellipse(ContinuousGeometryEntity):
    """Ellipse geometrical object"""

    def __init__(self, foci1: np.ndarray, foci2: np.ndarray, semi_major_axis: float):
        """Initialize a Ellipse geometrical object

        Args:
            foci1 (np.ndarray): first focal 2D point
            foci2 (np.ndarray): second focal 2D point
            semi_major_axis (float): semi major axis value
        """
        self.foci1 = foci1
        self.foci2 = foci2
        self.semi_major_axis = semi_major_axis  # also called "a" usually

    @property
    def centroid(self) -> np.ndarray:
        """Compute the center point of the ellipse

        Returns:
            np.ndarray: 2D point defining the center of the ellipse
        """
        return (self.foci1 + self.foci2) / 2

    @property
    def semi_minor_axis(self) -> float:
        """Computed semi minor axis (also called b usually)

        Returns:
            float: _description_
        """
        return math.sqrt(self.semi_major_axis**2 - self.linear_eccentricity**2)

    @property
    def linear_eccentricity(self) -> float:
        """Distance from any focal point to the center

        Returns:
            float: linear eccentricity value
        """
        return np.linalg.norm(self.foci2 - self.foci1) / 2

    @property
    def eccentricity(self) -> float:
        """Eccentricity value of the ellipse

        Returns:
            float: eccentricity value
        """
        return self.linear_eccentricity / self.semi_major_axis

    @property
    def h(self) -> float:
        """h is a common ellipse value used in calculation and kind of
        represents the eccentricity of the ellipse but in another perspective.

        Circle would have a h = 0. A really stretch out ellipse would have a h value
        close o 1

        Returns:
            float: h value
        """
        return (self.semi_major_axis - self.semi_minor_axis) ** 2 / (
            self.semi_major_axis + self.semi_minor_axis
        ) ** 2

    @property
    def area(self) -> float:
        return math.pi * self.semi_major_axis * self.semi_minor_axis

    def __perimeter_approx(self, n_terms: int = 5) -> float:
        """Perimeter approximation of the ellipse using the James Ivory
        infinite serie.

        See: https://en.wikipedia.org/wiki/Ellipse#Circumference

        Args:
            n_terms (int, optional): number of n first terms to calculate and
                add up from the infinite series. Defaults to 5.

        Returns:
            float: circumference approximation of the ellipse
        """
        sum = 1  # pre-calculated term n=0 equal 1
        for n in range(1, n_terms):  # goes from term n=1 to n=(n_terms-1)
            sum += (((1 / ((2 * n - 1) * (4**n))) * math.comb(2 * n, n)) ** 2) * (
                self.h**n
            )

        return math.pi * (self.semi_major_axis + self.semi_minor_axis) * sum

    @property
    def perimeter(self) -> float:
        """Compute the perimeter of the ellipse.
        Beware this is only an approximation due to the computation of both pi
        and the James Ivory's infinite serie.

        Returns:
            float: perimeter value
        """
        return self.__perimeter_approx()

    @property
    def curvature(self, point: np.ndarray) -> float:
        """Curvature at the point defined as parameter

        Args:
            point (np.ndarray): input point.

        Returns:
            float: _description_
        """
        x = point[0]
        y = point[1]
        return (1 / (self.semi_major_axis * self.semi_minor_axis) ** 2) * (
            (x**2 / self.semi_major_axis**4) + (y**2 / self.semi_minor_axis**4)
        ) ** (-3 / 2)

    def point_belongs(self, point: np.ndarray, error: Optional[float] = None) -> bool:
        if error is None:
            error = self.semi_minor_axis * (1 / 100)

        # TODO
