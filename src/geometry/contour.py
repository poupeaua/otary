"""
Contour class to handle complexity with contour calculation
"""

from __future__ import annotations

import copy
from abc import ABC
from typing import Optional, Self
import logging

import cv2
import matplotlib.pyplot as plt
import numpy as np
from shapely import LinearRing

from src.geometry import DEFAULT_MARGIN_ANGLE_ERROR
from src.geometry import GeometryEntity
from src.geometry import Segment


class ContourReducer(ABC):
    """
    Class to manage all the contour reducer methods.
    Contour reduction means identifying the edge points in a given contour that could
    be composed of tons of useless points.
    """

    REDUCE_BY_DISTANCE_UNSUCCESSIVE_MODES = ["keep_first", "mean"]

    @staticmethod
    def __reduce_collinear(
        points: np.ndarray, margin_error_angle: float = DEFAULT_MARGIN_ANGLE_ERROR
    ) -> np.ndarray:
        """Remove close collinear points in a list of points

        Args:
            points (np.ndarray): list of points as array of shape (n, 2)
            margin_error_angle (float, optional): minimum angle to suppress points.
                Defaults to DEFAULT_MARGIN_ANGLE_ERROR.

        Returns:
            (np.ndarray): points without any collinear close points
        """
        idx_to_remove = []
        for i, cur_point in enumerate(points):
            if i == len(points) - 1:
                i = -1  # so that next point is the first in that case
            elif i == len(points) - 2:
                i = -2
            next_point = points[i + 1]
            next_next_point = points[i + 2]
            seg1 = Segment(points=[cur_point, next_point])
            seg2 = Segment(points=[next_point, next_next_point])
            if seg1.is_collinear(segment=seg2, margin_error_angle=margin_error_angle):
                idx_to_remove.append(i + 1)

        points = np.delete(points, idx_to_remove, 0)
        return points

    @staticmethod
    def reduce_collinear(
        points: np.ndarray,
        margin_error_angle: float = DEFAULT_MARGIN_ANGLE_ERROR,
        n_iterations: int = 1,
    ) -> np.ndarray:
        """Iterative extension of the method named
        :func:`~ContourReducer.__reduce_collinear()`

        Args:
            points (np.ndarray): list of points as array of shape (n, 2)
            margin_error_angle (float, optional): minimum angle to suppress points.
                Defaults to DEFAULT_MARGIN_ANGLE_ERROR.
            n_iterations (int, optional): number of times to execute this reduce method

        Returns:
            (np.ndarray): points without any collinear points
        """
        for _ in range(n_iterations):
            n_pts_before = len(points)
            points = ContourReducer.__reduce_collinear(
                points=points, margin_error_angle=margin_error_angle
            )
            n_pts_after = len(points)
            if n_pts_before == n_pts_after:  # if no points change after reduce stop
                break

        return points

    @staticmethod
    def reduce_by_distance(
        points: np.ndarray, min_dist_threshold: float = 2
    ) -> np.ndarray:
        """Given a list of points, reduce the list by discarding the points that are
        close to each other.

        This reduce by distance function has a big drawback: it will remove
        potentially too much points between two given points A and B
        as long as there exists a suite of points that are close enough to each others
        between A and B. In order to avoid this disadvantage, please refer to the
        method named :func:`~ContourReducer.reduce_by_distance_unsuccessive`.

        Args:
            points (np.ndarray): list of points of shape (n, 2)
            min_dist_threshold (float, optional): if two points have a distance
                lower that this value, delete one point. Defaults to 2.

        Returns:
            np.ndarray: filtered list of points
        """
        idx_to_remove = []
        for i, cur_point in enumerate(points):
            if i == len(points) - 1:
                i = -1  # so that next point is the first in that case
            next_point = points[i + 1]
            distance = Segment(points=[cur_point, next_point]).length
            if distance < min_dist_threshold:
                idx_to_remove.append(i)

        reduced_points = np.delete(np.asarray(points), idx_to_remove, 0)

        return reduced_points

    @staticmethod
    def __reduce_by_distance_unsuccessive(
        points: np.ndarray,
        min_dist_threshold: float = 2,
        n: int = 2,
        mode: str = "keep_first",
    ) -> np.ndarray:
        """Given a list of points, reduces the list by discarding the points that are
        close to each other and by making sure that we limit the number of successive
        points that can be deleted.

        To understand why this approach is relevant, let us consider a contour that is
        formed by millions of millions of close points. Applying the basic method
        :func:`~ContourReducer.reduce_by_distance` would delete all points, which
        is not what we want. This method limits the number of deletion.

        Hence this method can be seen as an enhanced version of the function
        :func:`~ContourReducer.reduce_by_distance`.

        Args:
            points (np.ndarray): list of points of shape (n, 2)
            min_dist_threshold (float, optional): _description_. Defaults to 2.
            n: (int, optional): the minimum number of points to be detected as close
                before deleting and continue
            mode (str, optional): defines how the method handles the suppression of
                points.
                - If mode = 'keep_first', then we simply keep the first point
                all other successive points are deleted.
                - If mode = 'mean', the centroid point is calculated and replace all
                the other points to be suppressed.

        Returns:
            np.ndarray: filtered list of points
        """
        if mode not in ContourReducer.REDUCE_BY_DISTANCE_UNSUCCESSIVE_MODES:
            raise ValueError(
                f"The mode {mode} is not a valid mode. It should be in"
                f"{ContourReducer.REDUCE_BY_DISTANCE_UNSUCCESSIVE_MODES}"
            )
        idx_to_rm: list[int] = []
        cur_idx_to_rm: list[int] = []
        for i, cur_point in enumerate(points):
            if i == len(points) - 1:
                i = -1  # so that next point is the first in that case
            next_point = points[i + 1]
            distance = Segment(points=[cur_point, next_point]).length
            if distance < min_dist_threshold:
                if len(cur_idx_to_rm) == 0 or (
                    len(cur_idx_to_rm) > 0 and i == cur_idx_to_rm[-1] + 1
                ):
                    cur_idx_to_rm.append(i)
                else:
                    cur_idx_to_rm = []
            else:
                cur_idx_to_rm = []

            if len(cur_idx_to_rm) >= n:
                if mode == "mean":
                    first_idx_pt, last_idx_pt = (
                        cur_idx_to_rm[0],
                        cur_idx_to_rm[-1],
                    )
                    centroid_point = Contour(
                        points=points[first_idx_pt : (last_idx_pt + 1)]
                    ).centroid.astype(int)
                    points[first_idx_pt] = centroid_point
                idx_to_rm = idx_to_rm + cur_idx_to_rm[1:]  # all except first point idx
                cur_idx_to_rm = []

        points = np.delete(points, idx_to_rm, 0)
        return points

    @staticmethod
    def reduce_by_distance_unsuccessive(
        points: np.ndarray,
        min_dist_threshold: float = 2,
        n: int = 2,
        mode: str = "keep_first",
        n_iterations: int = 1,
    ) -> np.ndarray:
        """Iterative extension of the method
        :func:`~Contour._reduce_by_distance_unsuccessive()`

        Args:
            points (np.ndarray): list of points of shape (n, 2)
            min_dist_threshold (float, optional): _description_. Defaults to 2.
            n: (int, optional): the minimum number of points to be detected as close
                before deleting and continue
            mode (str, optional): defines how the method handles the suppression of
                points.
                - If mode = 'keep_first', then we simply keep the first point
                all other successive points are deleted.
                - If mode = 'mean', the centroid point is calculated and replace all
                the other points to be suppressed.
            n_iterations (int, optional): number of times to execute this reduce method

        Returns:
            np.ndarray: filtered list of points
        """
        for _ in range(n_iterations):
            n_pts_before = len(points)
            points = ContourReducer.__reduce_by_distance_unsuccessive(
                points=points, min_dist_threshold=min_dist_threshold, n=n, mode=mode
            )
            n_pts_after = len(points)
            if n_pts_before == n_pts_after:  # if no points change after reduce stop
                break

        return points

    @staticmethod
    def reduce_by_triangle_area(
        points: np.ndarray, min_triangle_area: float = 1
    ) -> np.ndarray:
        """This an implementation of the Visvalingam–Whyatt algorithm.
        An introduction for the algorithm can be found here:
        https://en.wikipedia.org/wiki/Visvalingam%E2%80%93Whyatt_algorithm

        The idea of the original algorithm is to conserve points in a contour
        construction which form, with its two closest neighbour points, a triangle with
        a sufficiently big area > A, having A as a parameter.
        The points which form a triangle area < A are descarded.

        Args:
            points (np.ndarray): _description_
            min_triangle_area (float, optional): _description_. Defaults to 1.

        Returns:
            (np.ndarray): points with a minimum triangle area
        """
        idx_to_remove = []
        for i, cur_point in enumerate(points):
            if i == len(points) - 1:
                i = -1  # so that next point is the first in that case
            prev_point, next_point = points[i - 1], points[i + 1]
            area = Contour(points=[prev_point, cur_point, next_point]).area
            if area < min_triangle_area:
                idx_to_remove.append(i)

        reduced_points = np.delete(np.asarray(points), idx_to_remove, 0)

        return reduced_points


class Contour(GeometryEntity, ContourReducer):
    """Contour class which defines a contour object of any closed-shape"""

    @property
    def shapely(self) -> LinearRing:
        """Returns the Shapely.LinearRing representation of the contour.
        See https://shapely.readthedocs.io/en/stable/reference/shapely.LinearRing.html

        Returns:
            LinearRing: shapely.LinearRing object
        """
        return LinearRing(coordinates=self.asarray)

    @property
    def lines(self) -> np.ndarray:
        """Expresses the Contour as a list of segments.

        Returns:
            np.ndarray: segments representation of the contour
        """
        return Contour.points_to_lines(self.points)

    @property
    def lengths(self) -> np.ndarray:
        """Returns the length of all the segments that make up the Contour

        Returns:
            np.ndarray: array of shape (n_points)
        """
        return np.linalg.norm(np.diff(self.lines, axis=1), axis=2)

    @property
    def is_self_intersected(self) -> bool:
        """Whether any of the segments intersect another segment in the same set

        Returns:
            bool: True if at least two lines intersect, False otherwise
        """
        return not self.shapely.is_simple

    # ---------------------------------- OTHER CONSTRUCTORS ----------------------------

    @classmethod
    def from_lines(cls, lines: np.ndarray) -> Contour:
        """The lines should describe a perfect closed shape contour

        Args:
            lines (np.ndarray): array of lines of shape (n, 2, 2)

        Returns:
            (Contour): a Contour object
        """
        nlines = len(lines)
        shifted_lines = np.roll(
            np.array(lines).reshape(nlines * 2, 2), shift=1, axis=0
        ).reshape(nlines, 2, 2)
        distances = np.linalg.norm(np.diff(shifted_lines, axis=1), axis=2)
        if np.any(distances):  # a distance is different from 0
            bad_idxs = np.nonzero(distances > 0)
            raise ValueError(
                f"Could not construct the contour from the given lines."
                f"Please check at those indices: {bad_idxs}"
            )
        contour_points = lines[:, 0]
        return Contour(points=contour_points)

    @classmethod
    def from_unordered_lines_approx(
        cls,
        lines: np.ndarray,
        max_dist_thresh: float = 50,
        max_iterations: int = 50,
        start_line_index: int = 0,
        img: Optional[np.ndarray] = None,
        is_debug_enabled: bool = False,
    ) -> Contour:
        # pylint: disable=too-many-positional-arguments,too-many-arguments
        """Create a Contour object from an unordered list of lines that approximate a
        closed-shape. They approximate in the sense that they do not necessarily
        share common points. This method computes the intersection points between lines.

        Args:
            img (_type_): array of shape (lx, ly)
            lines (np.ndarray): array of lines of shape (n, 2, 2)
            max_dist_thresh (float, optional): For any given point,
                the maximum distance to consider two points as close. Defaults to 50.
            max_iterations (float, optional): Maximum number of iterations before
                finding a contour.
                It defines also the maximum number of lines in the contour to find.
            start_line_index (int, optional): The starting line to find searching for
                the contour. Defaults to 0.

        Returns:
            (Contour): a Contour object
        """

        # pylint: disable=too-many-locals
        lines = np.asarray(lines)
        Segment.assert_list_of_lines(lines=lines)

        def debug_visualize(seg: np.ndarray):
            if is_debug_enabled and img is not None:
                im = img.copy()
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                im = cv2.line(
                    img=im, pt1=seg[0], pt2=seg[1], color=(0, 250, 126), thickness=5
                )
                plt.imshow(im)
                plt.xticks([])
                plt.yticks([])
                plt.show()

        _lines = copy.deepcopy(lines)
        list_build_cnt = []
        is_contour_found = False
        idx_seg_closest = start_line_index
        i = 0
        while not is_contour_found and i < max_iterations:
            curseg = Segment(_lines[idx_seg_closest])
            curpoint = curseg.asarray[1]
            list_build_cnt.append(curseg.asarray)
            _lines = np.delete(_lines, idx_seg_closest, axis=0)

            if len(_lines) == 0:
                logging.debug("No more lines to be processed.")

            # find the closest point to the current one and associated line
            lines2points = _lines.reshape(len(_lines) * 2, 2)
            dist_from_curpoint = np.linalg.norm(lines2points - curpoint, axis=1)
            idx_closest_points = np.nonzero(dist_from_curpoint < max_dist_thresh)[0]

            debug_visualize(seg=curseg.asarray)

            if len(idx_closest_points) > 1:
                # more than one point close to the current point - take the closest
                idx_closest_points = np.array([np.argmin(dist_from_curpoint)])
            if len(idx_closest_points) == 0:
                # no point detected - can mean that the contour is done or not
                first_seg = Segment(list_build_cnt[0])
                if np.linalg.norm(first_seg.asarray[0] - curpoint) < max_dist_thresh:
                    # TODO sometimes multiples intersection example 7
                    intersect_point = curseg.intersection_line(first_seg)
                    list_build_cnt[-1][1] = intersect_point
                    list_build_cnt[0][0] = intersect_point
                    is_contour_found = True
                    break
                raise RuntimeError("No point detected close to the current point")

            # only one closest point - get indices of unique closest point on segment
            idx_point_closest = int(idx_closest_points[0])
            idx_seg_closest = int(np.floor(idx_point_closest / 2))

            # arrange the line so that the closest point is in the first place
            idx_point_in_line = 0 if (idx_point_closest / 2).is_integer() else 1
            seg_closest = _lines[idx_seg_closest]
            if idx_point_in_line == 1:  # flip points positions
                seg_closest = np.flip(seg_closest, axis=0)
            _lines[idx_seg_closest] = seg_closest

            # find intersection point between the two lines
            intersect_point = curseg.intersection_line(Segment(seg_closest))

            # update arrays with the intersection point
            _lines[idx_seg_closest][0] = intersect_point
            list_build_cnt[i][1] = intersect_point

            i += 1

        cnt = Contour.from_lines(np.array(list_build_cnt))
        return cnt

    # ------------------------------ STATIC METHODS ------------------------------------

    @staticmethod
    def points_to_lines(points: np.ndarray) -> np.ndarray:
        """Static method to convert a contour described by points to lines

        Args:
            points (np.ndarray): array of points of shape (n, 2)

        Returns:
            np.ndarray: array of lines of shape (n, 2, 2)
        """
        return np.stack([points, np.roll(points, shift=-1, axis=0)], axis=1)

    # ------------------------------- CLASSIC METHODS ----------------------------------

    def is_regular(self, margin_dist_error_pct: float = 0.01) -> bool:
        """Identifies whether a contour is regular, this means is rectangular or is
        a square.

        Args:
            margin_area_error (float, optional): area error. Defaults to 25.

        Returns:
            bool: True if the contour describes a rectangle or square.
        """
        # check we have four points
        if len(self.asarray) != 4:
            return False

        # compute diagonal 1 = taking reference index as 1st point in list - index 0
        refpoint = self.asarray[0]
        idx_max_dist = self.index_farthest_point_from(point=refpoint)
        farther_point = self.asarray[idx_max_dist]
        diag1 = Segment(points=[refpoint, farther_point])

        # compute diagonal 2
        diag2_idxs = [1, 2, 3]  # every index except 0
        diag2_idxs.remove(idx_max_dist)  # delete index of point in first diag
        diag2 = Segment(points=self.asarray[diag2_idxs])

        # rectangular criteria = the diagonals have same lengths
        normed_length = np.sqrt(diag1.length * diag2.length)
        if np.abs(diag1.length - diag2.length) > normed_length * margin_dist_error_pct:
            return False

        # there should exist only one intersection point
        intersection_points = diag1.intersection(other=diag2)
        if len(intersection_points) != 1:
            return False

        # diagonals bisect on the center of both diagonal
        cross_point = intersection_points[0]
        dist_mid_cross_diag1 = np.linalg.norm(cross_point - diag1.centroid)
        dist_mid_cross_diag2 = np.linalg.norm(cross_point - diag2.centroid)
        if (
            np.abs(dist_mid_cross_diag1) > normed_length * margin_dist_error_pct
            or np.abs(dist_mid_cross_diag2) > normed_length * margin_dist_error_pct
        ):
            return False

        return True

    def score_contains_points(
        self, points: np.ndarray, min_distance: float
    ) -> np.ndarray:
        """Returns a score of 0 or 1 for each point in the contour if it is close
        enough to any point in the input points.

        Args:
            points (np.ndarray): list of 2D points
            margin_dist_error (float): mininum distance to consider two points as
                close enough to be considered as the same points

        Returns:
            np.ndarray: a list of score for each point in the contour
        """
        indices = self.indices_shared_close_points(
            other=Contour(points=points), margin_dist_error=min_distance
        )
        score = np.bincount(indices, minlength=len(self))
        return score

    # ---------------------------- MODIFICATION METHODS -------------------------------

    def add_point(self, point: np.ndarray, index: int) -> Self:
        """Add a point at a given index in the Contour object

        Args:
            point (np.ndarray): point to be added
            index (int): index where the point will be added

        Returns:
            Contour: Contour object with an added point
        """
        size = len(self)
        if index >= size:
            raise ValueError(
                f"The index value {index} is too big. "
                f"The maximum possible index value is {size-1}."
            )
        if index < 0:
            if abs(index) > size + 1:
                raise ValueError(
                    f"The index value {index} is too small. "
                    f"The minimum possible index value is {-(size+1)}"
                )
            index = size + index + 1

        self.points = np.concatenate(
            [self.points[:index], [point], self.points[index:]]
        )
        return self

    def rearrange_first_point_at_index(self, index: int) -> Self:
        """Rearrange the list of points that defines the Contour so that the first
        point in the list of points is the one at index given by the argument of this
        function.

        Args:
            index (int): index value

        Returns:
            Contour: Contour which is the exact same one but with a rearranged list
                of points.
        """
        size = len(self)
        if index >= size:
            raise ValueError(
                f"The index value {index} is too big. "
                f"The maximum possible index value is {size-1}."
            )
        if index < 0:
            if abs(index) > size:
                raise ValueError(
                    f"The index value {index} is too small. "
                    f"The minimum possible index value is {-size}"
                )
            index = size + index

        self.points = np.concatenate([self.points[index:], self.points[:index]])
        return self

    def rearrange_first_point_closest_to_reference_point(
        self, reference_point: np.ndarray = np.zeros(shape=(2,))
    ) -> Contour:
        """Rearrange the list of points that defines the Contour so that the first
        point in the list of points is the one that is the closest (by distance) to the
        reference point.

        Args:
            reference_point (np.ndarray): point that is taken as a reference in the
                space to find the one in the Contour list of points that is the
                closest to this reference point. Default to origin point [0, 0].

        Returns:
            Contour: Contour which is the exact same one but with a rearranged list
                of points.
        """
        idx_min_dist = self.index_closest_point_from(point=reference_point)
        return self.rearrange_first_point_at_index(index=idx_min_dist)

    # ------------------------------- Fundamental Methods ------------------------------

    def is_equal(self, contour: Contour, dist_margin_error: float = 5) -> bool:
        """Check whether two contours objects are equal by considering a margin of
        error based on a distance between points.

        Args:
            contour (Contour): Contour object
            dist_margin_error (float, optional): distance margin of error.
                Defaults to 5.

        Returns:
            bool: True if the contour are equal, False otherwise
        """
        if self.n_points != contour.n_points:
            # if contours do not have the same number of points they can not be similar
            return False

        # check if each points composing the contours are close to each other
        new_cnt = contour.copy().rearrange_first_point_closest_to_reference_point(
            self.points[0]
        )
        points_diff = new_cnt.points - self.points
        distances = np.linalg.norm(points_diff, axis=1)
        max_distance = np.max(distances)
        return max_distance <= dist_margin_error
