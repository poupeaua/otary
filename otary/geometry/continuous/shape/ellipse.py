"""
Ellipse Geometric Object
"""

from typing import Self, Optional

import math
import numpy as np
from numpy.typing import NDArray

from shapely import Polygon as SPolygon, LinearRing

from otary.geometry.utils.tools import rotate_2d_points
from otary.geometry.continuous.entity import ContinuousGeometryEntity
from otary.geometry import Polygon, Segment
from otary.utils.tools import assert_transform_shift_vector


class Ellipse(ContinuousGeometryEntity):
    """Ellipse geometrical object"""

    def __init__(
        self,
        foci1: NDArray,
        foci2: NDArray,
        semi_major_axis: float,
        n_points_polygonal_approx: int = ContinuousGeometryEntity.DEFAULT_N_POINTS_POLYGONAL_APPROX,
    ):
        """Initialize a Ellipse geometrical object

        Args:
            foci1 (NDArray): first focal 2D point
            foci2 (NDArray): second focal 2D point
            semi_major_axis (float): semi major axis value
            n_points_polygonal_approx (int, optional): number of points to be used in
                the polygonal approximation.
                Defaults to ContinuousGeometryEntity.DEFAULT_N_POINTS_POLYGONAL_APPROX.
        """
        super().__init__(n_points_polygonal_approx=n_points_polygonal_approx)
        self.foci1 = foci1
        self.foci2 = foci2
        self.semi_major_axis = semi_major_axis  # also called "a" usually
        self.__assert_ellipse()

        if type(self) == Ellipse:
            # to avoid computation in circle class instantiation
            # since the center attribute is not defined in the Circle class yet
            self.update_polyapprox()

    def __assert_ellipse(self) -> None:
        """Assert the parameters of the ellipse.
        If the parameters proposed do not make up a ellipse raise an error.
        """
        if self.semi_major_axis <= self.linear_eccentricity:
            raise ValueError(
                f"The semi major-axis (a={self.semi_major_axis}) can not be smaller "
                f"than the linear eccentricity (c={self.linear_eccentricity}). "
                "The ellipse is thus not valid."
            )

    # --------------------------------- PROPERTIES ------------------------------------

    @property
    def centroid(self) -> NDArray:
        """Compute the center point of the ellipse

        Returns:
            NDArray: 2D point defining the center of the ellipse
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
        return float(np.linalg.norm(self.foci2 - self.foci1) / 2)

    @property
    def focal_distance(self) -> float:
        """Distance from any focal point to the center

        Returns:
            float: focal distance value
        """
        return self.linear_eccentricity

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
        """Compute the area of the ellipse

        Returns:
            float: area value
        """
        return math.pi * self.semi_major_axis * self.semi_minor_axis

    @property
    def perimeter(self) -> float:
        """Compute the perimeter of the ellipse.
        Beware this is only an approximation due to the computation of both pi
        and the James Ivory's infinite serie.

        Returns:
            float: perimeter value
        """
        return self.perimeter_approx()

    def curvature(self, point: NDArray) -> float:
        """Curvature at the point defined as parameter

        Args:
            point (NDArray): input point.

        Returns:
            float: _description_
        """
        x = point[0]
        y = point[1]
        return (1 / (self.semi_major_axis * self.semi_minor_axis) ** 2) * (
            (x**2 / self.semi_major_axis**4) + (y**2 / self.semi_minor_axis**4)
        ) ** (-3 / 2)

    @property
    def shapely_surface(self) -> SPolygon:
        """Returns the Shapely.Polygon as an surface representation of the Ellipse.
        See https://shapely.readthedocs.io/en/stable/reference/shapely.Polygon.html

        Returns:
            Polygon: shapely.Polygon object
        """
        return SPolygon(self.polyaprox.asarray, holes=None)

    @property
    def shapely_edges(self) -> LinearRing:
        """Returns the Shapely.LinearRing as a curve representation of the Ellipse.
        See https://shapely.readthedocs.io/en/stable/reference/shapely.LinearRing.html

        Returns:
            LinearRing: shapely.LinearRing object
        """
        return LinearRing(coordinates=self.polyaprox.asarray)

    # ---------------------------- MODIFICATION METHODS -------------------------------

    def rotate(
        self,
        angle: float,
        is_degree: bool = False,
        is_clockwise: bool = True,
        pivot: Optional[NDArray] = None,
    ) -> Self:
        """Rotate the ellipse around a pivot point.

        Args:
            angle (float): angle to rotate the ellipse
            is_degree (bool, optional): whether the angle is in degrees. Defaults to False.
            is_clockwise (bool, optional): whether the rotation is clockwise. Defaults to True.
            pivot (Optional[NDArray], optional): pivot point to rotate around. Defaults to None.

        Returns:
            Self: rotated ellipse object
        """
        if is_degree:
            angle = math.radians(angle)
        if is_clockwise:
            angle = -angle

        if pivot is None:
            pivot = self.centroid

        self.foci1 = rotate_2d_points(self.foci1, angle, pivot)
        self.foci2 = rotate_2d_points(self.foci2, angle, pivot)
        self.update_polyapprox()
        return self

    def shift(self, vector: NDArray) -> Self:
        """Shift the ellipse by a given vector.

        Args:
            vector (NDArray): vector to shift the ellipse

        Returns:
            Self: shifted ellipse object
        """
        assert_transform_shift_vector(vector)
        self.foci1 += vector
        self.foci2 += vector
        self.update_polyapprox()
        return self

    def normalize(self, x: float, y: float) -> Self:
        """Normalize the ellipse to a given bounding box.

        Args:
            x (float): width of the bounding box
            y (float): height of the bounding box

        Returns:
            Self: normalized ellipse object
        """
        factor = np.array([x, y])
        self.foci1 = self.foci1 / factor
        self.foci2 = self.foci2 / factor

        self.update_polyapprox()
        return self

    # ------------------------------- CLASSIC METHODS ---------------------------------

    def perimeter_approx(self, n_terms: int = 5, is_ramanujan: bool = False) -> float:
        """Perimeter approximation of the ellipse using the James Ivory
        infinite serie. In the case of the circle this always converges to the
        exact value of the circumference no matter the number of terms.

        See: https://en.wikipedia.org/wiki/Ellipse#Circumference

        Args:
            n_terms (int, optional): number of n first terms to calculate and
                add up from the infinite series. Defaults to 5.
            is_ramanujan (bool, optional): whether to use the Ramanujan's best
                approximation.

        Returns:
            float: circumference approximation of the ellipse
        """
        if is_ramanujan:
            return (
                math.pi
                * (self.semi_major_axis + self.semi_minor_axis)
                * (1 + (3 * self.h) / (10 + math.sqrt(4 - 3 * self.h)))
            )

        sum = 1  # pre-calculated term n=0 equal 1
        for n in range(1, n_terms):  # goes from term n=1 to n=(n_terms-1)
            sum += (((1 / ((2 * n - 1) * (4**n))) * math.comb(2 * n, n)) ** 2) * (
                self.h**n
            )

        return math.pi * (self.semi_major_axis + self.semi_minor_axis) * sum

    def polygonal_approx(self, n_points: int, is_cast_int: bool = False) -> Polygon:
        """Generate apolygonal approximation of the ellipse.

        The way is done is the following:
        1. suppose the ellipse centered at the origin
        2. suppose the ellipse semi major axis to be parallel with the x-axis
        3. compute pairs of (x, y) points that belong to the ellipse using the
            parametric equation of the ellipse.
        4. shift all points by the same shift as the center to origin
        5. rotate using the ellipse center pivot point

        Args:
            n_points (int): number of points that make up the ellipse
                polygonal approximation
            is_cast_int (bool): whether to cast to int the points coordinates or
                not. Defaults to False

        Returns:
            Polygon: Polygon representing the ellipse as a succession of n points
        """
        points = []
        for theta in np.linspace(0, 2 * math.pi, n_points):
            x = self.semi_major_axis * math.cos(theta)
            y = self.semi_minor_axis * math.sin(theta)
            points.append([x, y])

        poly = Polygon(points=np.asarray(points), is_cast_int=False)
        poly.shift(vector=self.centroid)
        angle = Segment(points=[self.foci1, self.foci2]).slope_angle()
        poly.rotate(angle=angle)

        if is_cast_int:
            poly.asarray = poly.asarray.astype(int)

        return poly

    def copy(self) -> Self:
        """Copy the current ellipse object

        Returns:
            Self: copied ellipse object
        """
        return type(self)(
            foci1=self.foci1,
            foci2=self.foci2,
            semi_major_axis=self.semi_major_axis,
            n_points_polygonal_approx=self.n_points_polygonal_approx,
        )

    def __str__(self) -> str:
        return (
            f"Ellipse(foci1={self.foci1}, foci2={self.foci2}, a={self.semi_major_axis})"
        )

    def __repr__(self) -> str:
        return (
            f"Ellipse(foci1={self.foci1}, foci2={self.foci2}, a={self.semi_major_axis})"
        )
