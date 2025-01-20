"""
Polygon class to handle complexity with polygon calculation
"""

from __future__ import annotations

from abc import ABC
import copy
from typing import Optional, Self
import logging

import cv2
import matplotlib.pyplot as plt
import numpy as np
from shapely import LinearRing, Polygon as SPolygon

from src.geometry.entity import GeometryEntity
from src.geometry.discrete.entity import DiscreteGeometryEntity
from src.geometry import Segment, Vector, DEFAULT_MARGIN_ANGLE_ERROR


class PolygonReducer(ABC):
    """Class to manage all the polygon reducer methods.

    Polygon reduction means identifying the edge points in a given polygon that could
    be composed of tons of useless points and deleting these points. In the end,
    it makes the polygons having less points but describing the same shape.
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
        :func:`~PolygonReducer.__reduce_collinear()`

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
            points = PolygonReducer.__reduce_collinear(
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
        method named :func:`~PolygonReducer.reduce_by_distance_unsuccessive`.

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

        To understand why this approach is relevant, let us consider a polygon that is
        formed by millions of millions of close points. Applying the basic method
        :func:`~PolygonReducer.reduce_by_distance` would delete all points, which
        is not what we want. This method limits the number of deletion.

        Hence this method can be seen as an enhanced version of the function
        :func:`~PolygonReducer.reduce_by_distance`.

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
        if mode not in PolygonReducer.REDUCE_BY_DISTANCE_UNSUCCESSIVE_MODES:
            raise ValueError(
                f"The mode {mode} is not a valid mode. It should be in"
                f"{PolygonReducer.REDUCE_BY_DISTANCE_UNSUCCESSIVE_MODES}"
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
                    centroid_point = Polygon(
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
        :func:`~Polygon._reduce_by_distance_unsuccessive()`

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
            points = PolygonReducer.__reduce_by_distance_unsuccessive(
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

        The idea of the original algorithm is to conserve points in a polygon
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
            area = Polygon(points=[prev_point, cur_point, next_point]).area
            if area < min_triangle_area:
                idx_to_remove.append(i)

        reduced_points = np.delete(np.asarray(points), idx_to_remove, 0)

        return reduced_points


class Polygon(DiscreteGeometryEntity, PolygonReducer):
    """Polygon class which defines a polygon object which means any closed-shape"""

    # ---------------------------------- OTHER CONSTRUCTORS ----------------------------

    @classmethod
    def from_lines(cls, lines: np.ndarray) -> Polygon:
        """The lines should describe a perfect closed shape polygon

        Args:
            lines (np.ndarray): array of lines of shape (n, 2, 2)

        Returns:
            (Polygon): a Polygon object
        """
        nlines = len(lines)
        shifted_lines = np.roll(
            np.array(lines).reshape(nlines * 2, 2), shift=1, axis=0
        ).reshape(nlines, 2, 2)
        distances = np.linalg.norm(np.diff(shifted_lines, axis=1), axis=2)
        if np.any(distances):  # a distance is different from 0
            bad_idxs = np.nonzero(distances > 0)
            raise ValueError(
                f"Could not construct the polygon from the given lines."
                f"Please check at those indices: {bad_idxs}"
            )
        points = lines[:, 0]
        return Polygon(points=points)

    @classmethod
    def from_unordered_lines_approx(
        cls,
        lines: np.ndarray,
        max_dist_thresh: float = 50,
        max_iterations: int = 50,
        start_line_index: int = 0,
        img: Optional[np.ndarray] = None,
        is_debug_enabled: bool = False,
    ) -> Polygon:
        # pylint: disable=too-many-positional-arguments,too-many-arguments
        """Create a Polygon object from an unordered list of lines that approximate a
        closed-shape. They approximate in the sense that they do not necessarily
        share common points. This method computes the intersection points between lines.

        Args:
            img (_type_): array of shape (lx, ly)
            lines (np.ndarray): array of lines of shape (n, 2, 2)
            max_dist_thresh (float, optional): For any given point,
                the maximum distance to consider two points as close. Defaults to 50.
            max_iterations (float, optional): Maximum number of iterations before
                finding a polygon.
                It defines also the maximum number of lines in the polygon to be found.
            start_line_index (int, optional): The starting line to find searching for
                the polygon. Defaults to 0.

        Returns:
            (Polygon): a Polygon object
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
        is_polygon_found = False
        idx_seg_closest = start_line_index
        i = 0
        while not is_polygon_found and i < max_iterations:
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
                # no point detected - can mean that the polygon is done or not
                first_seg = Segment(list_build_cnt[0])
                if np.linalg.norm(first_seg.asarray[0] - curpoint) < max_dist_thresh:
                    # TODO sometimes multiples intersection example 7
                    intersect_point = curseg.intersection_line(first_seg)
                    list_build_cnt[-1][1] = intersect_point
                    list_build_cnt[0][0] = intersect_point
                    is_polygon_found = True
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

        cnt = Polygon.from_lines(np.array(list_build_cnt))
        return cnt

    # --------------------------------- PROPERTIES ------------------------------------

    @property
    def shapely_surface(self) -> SPolygon:
        """Returns the Shapely.Polygon as an surface representation of the Polygon.
        See https://shapely.readthedocs.io/en/stable/reference/shapely.Polygon.html

        Returns:
            Polygon: shapely.Polygon object
        """
        return SPolygon(self.asarray, holes=None)

    @property
    def shapely_edges(self) -> LinearRing:
        """Returns the Shapely.LinearRing as a curve representation of the Polygon.
        See https://shapely.readthedocs.io/en/stable/reference/shapely.LinearRing.html

        Returns:
            LinearRing: shapely.LinearRing object
        """
        return LinearRing(coordinates=self.asarray)

    @property
    def segments(self) -> np.ndarray:
        """Describes the Polygon as a list of segments.

        Returns:
            np.ndarray: segments representation of the polygon
        """
        return Polygon.points_to_segments(self.points)

    @property
    def lengths(self) -> np.ndarray:
        """Returns the length of all the segments that make up the Polygon

        Returns:
            np.ndarray: array of shape (n_points)
        """
        return np.linalg.norm(np.diff(self.segments, axis=1), axis=2)

    @property
    def is_self_intersected(self) -> bool:
        """Whether any of the segments intersect another segment in the same set

        Returns:
            bool: True if at least two lines intersect, False otherwise
        """
        return not self.shapely_edges.is_simple

    @property
    def is_convex(self) -> bool:
        """Whether the Polygon describes a convex shape of not.

        Returns:
            bool: True if convex else False
        """
        return cv2.isContourConvex(contour=self.asarray)

    # ------------------------------ STATIC METHODS ------------------------------------

    @staticmethod
    def points_to_segments(points: np.ndarray) -> np.ndarray:
        """Static method to convert a polygon described by points to lines

        Args:
            points (np.ndarray): array of points of shape (n, 2)

        Returns:
            np.ndarray: array of lines of shape (n, 2, 2)
        """
        return np.stack([points, np.roll(points, shift=-1, axis=0)], axis=1)

    # ------------------------------- CLASSIC METHODS ----------------------------------

    def is_regular(self, margin_dist_error_pct: float = 0.01) -> bool:
        """Identifies whether the polygon is regular, this means is rectangular or is
        a square.

        Args:
            margin_area_error (float, optional): area error. Defaults to 25.

        Returns:
            bool: True if the polygon describes a rectangle or square.
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

    def contains(self, other: GeometryEntity, dilate_scale: float = 1) -> bool:
        """Whether the geometry contains the other or not

        Args:
            other (GeometryEntity): a GeometryEntity object
            dilate_scale (float): if greater than 1, the object will be scaled up
                before checking if it contains the other Geometry Entity. Can not be
                a value less than 1.

        Returns:
            bool: True if the entity contains the other
        """
        surface = self.shapely_surface
        if dilate_scale > 1:
            surface = self.copy().expand(scale=dilate_scale).shapely_surface
        return surface.contains(other.shapely_surface)

    def score_contains_points(
        self, points: np.ndarray, min_distance: float
    ) -> np.ndarray:
        """Returns a score of 0 or 1 for each point in the polygon if it is close
        enough to any point in the input points.

        Args:
            points (np.ndarray): list of 2D points
            margin_dist_error (float): mininum distance to consider two points as
                close enough to be considered as the same points

        Returns:
            np.ndarray: a list of score for each point in the contour
        """
        indices = self.indices_shared_approx_points(
            other=Polygon(points=points), margin_dist_error=min_distance
        )
        score = np.bincount(indices, minlength=len(self))
        return score

    # ---------------------------- MODIFICATION METHODS -------------------------------

    def add_point(self, point: np.ndarray, index: int) -> Self:
        """Add a point at a given index in the Polygon object

        Args:
            point (np.ndarray): point to be added
            index (int): index where the point will be added

        Returns:
            Polygon: Polygon object with an added point
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
        """Rearrange the list of points that defines the Polygon so that the first
        point in the list of points is the one at index given by the argument of this
        function.

        Args:
            index (int): index value

        Returns:
            Polygon: Polygon which is the exact same one but with a rearranged list
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
    ) -> Polygon:
        """Rearrange the list of points that defines the Polygon so that the first
        point in the list of points is the one that is the closest (by distance) to the
        reference point.

        Args:
            reference_point (np.ndarray): point that is taken as a reference in the
                space to find the one in the Polygon list of points that is the
                closest to this reference point. Default to origin point [0, 0].

        Returns:
            Polygon: Polygon which is the exact same one but with a rearranged list
                of points.
        """
        idx_min_dist = self.index_closest_point_from(point=reference_point)
        return self.rearrange_first_point_at_index(index=idx_min_dist)

    def __rescale(self, scale: float) -> Polygon:
        """Create a new polygon that is scaled up or down.

        The rescale method compute the vector that is directed from the polygon center
        to each point. Then it rescales each vector and use the head point of each
        vector to compose the new scaled polygon.

        Args:
            scale (float): float value to scale the polygon

        Returns:
            Polygon: scaled polygon
        """
        center = self.centroid
        for i, point in enumerate(self.asarray):
            self.asarray[i] = Vector([center, point]).rescale_head(scale).head
        return self

    def expand(self, scale: float) -> Polygon:
        """Stretch, dilate or expand a polygon

        Args:
            scale (float): scale expanding factor. Must be greater than 1.

        Returns:
            Polygon: new bigger polygon
        """
        if scale < 1:
            raise ValueError(
                "The scale value can not be less than 1 when expanding a polygon. "
                f"Found {scale}"
            )
        return self.__rescale(scale=scale)

    def shrink(self, scale: float) -> Polygon:
        """Contract or shrink a polygon

        Args:
            scale (float): scale shrinking factor. Must be greater than 1.

        Returns:
            Polygon: new bigger polygon
        """
        if scale < 1:
            raise ValueError(
                "The scale value can not be less than 1 when shrinking a polygon. "
                f"Found {scale}"
            )
        return self.__rescale(scale=1 / scale)

    # ------------------------------- Fundamental Methods ------------------------------

    def is_equal(self, polygon: Polygon, dist_margin_error: float = 5) -> bool:
        """Check whether two polygons objects are equal by considering a margin of
        error based on a distance between points.

        Args:
            polygon (Polygon): Polygon object
            dist_margin_error (float, optional): distance margin of error.
                Defaults to 5.

        Returns:
            bool: True if the polygon are equal, False otherwise
        """
        if self.n_points != polygon.n_points:
            # if polygons do not have the same number of points they can not be similar
            return False

        # check if each points composing the polygons are close to each other
        new_cnt = polygon.copy().rearrange_first_point_closest_to_reference_point(
            self.points[0]
        )
        points_diff = new_cnt.points - self.points
        distances = np.linalg.norm(points_diff, axis=1)
        max_distance = np.max(distances)
        return max_distance <= dist_margin_error
