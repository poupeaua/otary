"""
Segment class to describe defined lines and segments
"""

from __future__ import annotations

import itertools

import numpy as np
from sympy.geometry import Line

from src.geometry.utils.constants import DEFAULT_MARGIN_ANGLE_ERROR
from src.geometry.discrete.linear.entity import LinearEntity


class Segment(LinearEntity):
    """Segment Class to manipulate easily segments objects"""

    def __init__(self, points: np.ndarray | list, is_cast_int: bool = False) -> None:
        assert len(points) == 2
        assert len(points[0]) == 2
        assert len(points[1]) == 2
        super().__init__(points=points, is_cast_int=is_cast_int)

    @property
    def centroid(self) -> np.ndarray:
        """Returns the center point of the segment

        Returns:
            np.ndarray: point of shape (1, 2)
        """
        return np.sum(self.points, axis=0) / 2

    @property
    def slope(self) -> float:
        """Returns the segment slope in the classical XY coordinates referential

        Returns:
            float: segment slope value
        """
        p1, p2 = self.points[0], self.points[1]
        try:
            slope = (p2[1] - p1[1]) / (p2[0] - p1[0] + 1e-9)
        except ZeroDivisionError:
            slope = np.inf
        return slope

    @property
    def slope_cv2(self) -> float:
        """Compute the slope seen as in the cv2 coordinates with y-axis inverted

        Returns:
            float: segment slope value
        """
        return -self.slope

    @staticmethod
    def assert_list_of_lines(lines: np.ndarray) -> None:
        """Check that the lines argument is really a list of lines

        Args:
            lines (np.ndarray): a expected list of lines
        """
        if lines.shape[1:] != (2, 2):
            raise ValueError(
                "The input segments argument has not the expected shape. "
                f"Input shape {lines.shape[1:]}, expected shape (2, 2)."
            )

    def slope_angle(self, degree: bool = False, is_cv2: bool = False) -> float:
        """Calculate the slope angle of a single line in the cartesian space

        Args:
            degree (bool): whether to output the result in degree. By default in radian.

        Returns:
            float: slope angle in ]-pi/2, pi/2[
        """
        angle = np.arctan(self.slope_cv2) if is_cv2 else np.arctan(self.slope)
        if degree:
            angle = np.rad2deg(angle)
        return angle

    def is_parallel(
        self, segment: Segment, margin_error_angle: float = DEFAULT_MARGIN_ANGLE_ERROR
    ) -> bool:
        """Check if two lines are parallel by calculating the slope of the two lines

        Angle Difference = |theta_0 - theta_1| mod pi
        Because always returns positive results due to the modulo we took into account
        the special case where angle difference = np.pi - epsilon ~ 3.139,
        this implies also two parralel lines.

        Args:
            segment (np.array): segment of shape (2, 2)
            margin_error_angle (float, optional): Threshold value for validating
                if the lines are parallel. Defaults to DEFAULT_MARGIN_ANGLE_ERROR.

        Returns:
            bool: whether we qualify the lines as parallel or not
        """
        angle_difference = np.mod(
            np.abs(self.slope_angle() - segment.slope_angle()), np.pi
        )
        test = bool(
            angle_difference < margin_error_angle
            or np.abs(angle_difference - np.pi) < margin_error_angle
        )
        return test

    @staticmethod
    def is_points_collinear(
        p1: np.ndarray,
        p2: np.ndarray,
        p3: np.ndarray,
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
            segment=segment2, margin_error_angle=margin_error_angle
        )

    def is_point_collinear(
        self,
        point: np.ndarray,
        margin_error_angle: float = DEFAULT_MARGIN_ANGLE_ERROR,
    ) -> bool:
        """Check whether a point is collinear with the segment

        Args:
            point (np.ndarray): point of shape (2,)
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

        segment1 = Segment([[339, 615], [564, 650]])
        segment2 = Segment([[340, 614], [611, 657]])
        segment1.is_collinear(segment2)
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
            segment=segment, margin_error_angle=margin_error_angle
        )
        _is_collinear = 1 in val_arr
        return bool(_is_parallel and _is_collinear)

    def intersection_line(self, other: Segment) -> np.ndarray:
        """Compute the intersection point that would exist between two segments if we
        consider them as lines - which means as lines with infinite length.

        Lines would thus define infinite extension in both extremities directions
        of the input segments objects.

        Args:
            other (Segment): other Segment object

        Returns:
            np.ndarray: intersection point between the two lines
        """
        if self.is_parallel(segment=other, margin_error_angle=0):
            return np.array([])
        line0 = Line(self.asarray[0], self.asarray[1])
        line1 = Line(other.asarray[0], other.asarray[1])
        intersection = np.array(line0.intersection(line1)[0].evalf(n=7))
        return intersection
