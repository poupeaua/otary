"""
Segment class to describe defined lines and segments
"""

from __future__ import annotations

import math
import itertools
from typing import TYPE_CHECKING
import warnings

import numpy as np
from numpy.typing import NDArray

from otary.geometry.utils.constants import DEFAULT_MARGIN_ANGLE_ERROR
from otary.geometry.discrete.linear.entity import LinearEntity

if TYPE_CHECKING:  # pragma: no cover
    from typing_extensions import Self
else:  # pragma: no cover
    try:
        from typing import Self
    except ImportError:  # make Self available in Python <= 3.10
        from typing_extensions import Self


class Segment(LinearEntity):
    """Segment class"""

    def __init__(self, points: NDArray | list, is_cast_int: bool = False) -> None:
        assert len(points) == 2
        assert len(points[0]) == 2
        assert len(points[1]) == 2
        super().__init__(points=points, is_cast_int=is_cast_int)

    @property
    def centroid(self) -> NDArray:
        """Returns the center point of the segment

        Returns:
            NDArray: point of shape (1, 2)
        """
        return np.sum(self.points, axis=0) / 2

    @property
    def midpoint(self) -> NDArray:
        """In the Segment, this is equivalent to the centroid

        Returns:
            NDArray: point of shape (1, 2)
        """
        return self.centroid

    @property
    def direction_vector(self) -> NDArray:
        """Returns the direction vector of the segment from point 1 to point 2

        Returns:
            NDArray: direction vector of shape (2,)
        """
        return self.points[1] - self.points[0]

    @property
    def slope(self) -> float:
        """Returns the segment slope in the classical XY coordinates referential

        Can return inf if you have a really specific vertical line of the form:

        >>> seg = ot.Segment([[1e-9, 0], [0, 1]])
        >>> seg.slope
        inf

        Returns:
            float: segment slope value
        """
        p1, p2 = self.points[0], self.points[1]
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=RuntimeWarning)
            try:
                slope = (p2[1] - p1[1]) / (p2[0] - p1[0] + 1e-9)
            except RuntimeWarning:  # Now this is raised as an exception
                slope = np.inf
        return slope

    @property
    def slope_cv2(self) -> float:
        """Compute the slope seen as in the cv2 coordinates with y-axis inverted

        Returns:
            float: segment slope value
        """
        return -self.slope

    def slope_angle(self, degree: bool = False, is_y_axis_down: bool = False) -> float:
        """Calculate the slope angle of a single line in the cartesian space

        Args:
            degree (bool): whether to output the result in degree. By default in radian.

        Returns:
            float: slope angle in ]-pi/2, pi/2[
        """
        angle = np.arctan(self.slope_cv2) if is_y_axis_down else np.arctan(self.slope)
        if degree:
            angle = np.rad2deg(angle)
        return angle

    def is_parallel(
        self, other: Segment, margin_error_angle: float = DEFAULT_MARGIN_ANGLE_ERROR
    ) -> bool:
        """Check if two lines are parallel by calculating the slope of the two lines

        Angle Difference = |theta_0 - theta_1| mod pi
        Because always returns positive results due to the modulo we took into account
        the special case where angle difference = pi - epsilon ~ 3.139,
        this implies also two parallel lines.

        Args:
            segment (np.array): segment of shape (2, 2)
            margin_error_angle (float, optional): Threshold value for validating
                if the lines are parallel. Defaults to DEFAULT_MARGIN_ANGLE_ERROR.

        Returns:
            bool: whether we qualify the lines as parallel or not
        """
        if margin_error_angle == 0:
            # no margin of error, strict equality
            M = np.array([self.direction_vector, -other.direction_vector]).T  # (2,2)
            # Check if matrix is singular (parallel lines)
            cond = np.linalg.det(M) == 0
        else:
            # with margin of error, we use angle difference
            angle_difference = np.mod(
                np.abs(self.slope_angle() - other.slope_angle()), math.pi
            )
            cond = bool(
                angle_difference <= margin_error_angle
                or np.abs(angle_difference - math.pi) <= margin_error_angle
            )
        return cond

    @staticmethod
    def is_points_collinear(
        p1: NDArray,
        p2: NDArray,
        p3: NDArray,
        margin_error_angle: float = DEFAULT_MARGIN_ANGLE_ERROR,
    ) -> bool:
        """Verify whether three points on the plane are collinear or not.
        Method by angle or slope: For three points, slope of any pair of points must
        be same as other pair.

        Args:
            p1 (np.array): point of shape (2,)
            p2 (np.array): point of shape (2,)
            p3 (np.array): point of shape (2,)
            margin_error_angle (float, optional): Threshold value for validating
                collinearity. Defaults to DEFAULT_MARGIN_ANGLE_ERROR.

        Returns:
            bool: 1 if colinear, 0 otherwise
        """
        p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)

        # 2 or 3 points equal
        if (
            not np.logical_or(*(p1 - p2))
            or not np.logical_or(*(p1 - p3))
            or not np.logical_or(*(p2 - p3))
        ):
            return True

        segment1, segment2 = Segment([p1, p2]), Segment([p1, p3])
        return segment1.is_parallel(
            other=segment2, margin_error_angle=margin_error_angle
        )

    def is_point_collinear(
        self,
        point: NDArray,
        margin_error_angle: float = DEFAULT_MARGIN_ANGLE_ERROR,
    ) -> bool:
        """Check whether a point is collinear with the segment

        Args:
            point (NDArray): point of shape (2,)
            margin_error_angle (float, optional): Threshold value for validating
                collinearity. Defaults to DEFAULT_MARGIN_ANGLE_ERROR.

        Returns:
            bool: True if the point is collinear with the segment
        """
        return self.is_points_collinear(
            p1=self.asarray[0],
            p2=self.asarray[1],
            p3=point,
            margin_error_angle=margin_error_angle,
        )

    def is_collinear(
        self, segment: Segment, margin_error_angle: float = DEFAULT_MARGIN_ANGLE_ERROR
    ) -> bool:
        """Verify whether two segments on the plane are collinear or not.
        This means that they are parallel and have at least three points in common.

        We needed to make all the combination verification in order to proove cause we
        could end up with two points very very close by and it would end up not
        providing the expected result. Consider the following example:

        >>> segment1 = Segment([[339, 615], [564, 650]])
        >>> segment2 = Segment([[340, 614], [611, 657]])
        >>> segment1.is_collinear(segment2)
        Angle difference: 0.9397169393235674 Margin: 0.06283185307179587
        False

        Only because [339, 615] and [340, 614] are really close and do not provide the
        appropriate slope does not means that overall the two segments are not
        collinear.

        Args:
            segment (np.array): segment of shape (2, 2)
            margin_error_angle (float, optional): Threshold value for validating
                collinearity.

        Returns:
            bool: 1 if colinear, 0 otherwise
        """
        cur2lines = np.array([self.asarray, segment.asarray])
        points = np.concatenate(cur2lines, axis=0)
        val_arr = np.zeros(shape=4)
        for i, combi in enumerate(
            itertools.combinations(np.linspace(0, 3, 4, dtype=int), 3)
        ):
            val_arr[i] = Segment.is_points_collinear(
                p1=points[combi[0]],
                p2=points[combi[1]],
                p3=points[combi[2]],
                margin_error_angle=margin_error_angle,
            )

        _is_parallel = self.is_parallel(
            other=segment, margin_error_angle=margin_error_angle
        )
        _is_collinear = 1 in val_arr
        return bool(_is_parallel and _is_collinear)

    def intersection_line(self, other: Segment) -> NDArray:
        """Compute the intersection point that would exist between two segments if we
        consider them as lines - which means as lines with infinite length.

        Lines would thus define infinite extension in both extremities directions
        of the input segments objects.

        Args:
            other (Segment): other Segment object

        Returns:
            NDArray: intersection point between the two lines
        """
        if self.is_parallel(other, margin_error_angle=0):
            return np.array([])

        M = np.array([self.direction_vector, -other.direction_vector]).T  # shape (2,2)
        b = other.asarray[0] - self.asarray[0]  # shape (2,)
        solution, _ = np.linalg.solve(M, b)

        intersection = self.asarray[0] + solution * self.direction_vector
        return intersection

    def normal(self) -> Self:
        """
        Returns the normal segment of the segment.
        The normal segment is a segment that is orthogonal to the input segment.

        Please note that the normal segment have the same length as the input segment.
        Moreover the normal segment is rotated by 90 degrees clockwise.

        Returns:
            Segment: normal segment centered at the original segment centroid
        """
        normal = self.copy().rotate(
            angle=math.pi / 2, is_degree=False, is_clockwise=True
        )
        return normal
