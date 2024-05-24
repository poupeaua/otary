"""
Contour class to handle complexity with contour calculation
"""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np
from sympy.geometry import Line

from src.geometry import DEFAULT_MARGIN_ANGLE_ERROR
from src.geometry import GeometryEntity
from src.geometry import Segment


class Contour(GeometryEntity):
    def __init__(self, points: np.ndarray | list, reduce: bool = False) -> None:
        super().__init__(points)

    @property
    def n_points(self) -> int:
        return len(self.points)

    @property
    def lines(self) -> np.ndarray:
        return Contour.points_to_lines(self.points)

    @property
    def lengths(self) -> np.ndarray:
        return np.linalg.norm(np.diff(self.lines, axis=1), axis=2)

    # ---------------------------------- OTHER CONSTRUCTORS --------------------------------------

    @classmethod
    def from_lines(cls, lines: np.ndarray) -> Contour:
        """The lines should describe a perfect closed shape contour

        Args:
            lines (np.ndarray): array of lines of shape (n, 2, 2)

        Returns:
            (Contour): a Contour object
        """
        # assert lines quality
        nlines = len(lines)
        shifted_lines = np.roll(
            np.array(lines).reshape(nlines * 2, 2), shift=1, axis=0
        ).reshape(nlines, 2, 2)
        distances = np.linalg.norm(np.diff(shifted_lines, axis=1), axis=2)
        if np.any(distances):
            # a distance is different from 0
            bad_idxs = np.nonzero(distances > 0)
            raise RuntimeError(
                f"Could not construct the contour from the given lines."
                f"Please check at those indices: {bad_idxs}"
            )

        contour_points = lines[:, 0]
        return Contour(points=contour_points)

    @classmethod
    def from_unordered_lines_approx(
        cls,
        lines: np.ndarray,
        min_dist_threshold: float = 50,
        max_iteration: int = 50,
        start_line_index: int = 0,
        img: Optional[np.ndarray] = None,
        debug: bool = False,
    ) -> Contour:
        """Create a Contour object from an unordered list of lines that approximate a
        closed-shape. They approximate in the sense that they do not necessarily
        share common points. This method computes the intersection points between lines.

        Args:
            img (_type_): array of shape (lx, ly)
            lines (np.ndarray): array of lines of shape (n, 2, 2)
            min_dist_threshold (float, optional): For any given point,
                the minimum distance . Defaults to 50.
            max_iteration (float, optional): Maximum number of iterations before
                finding a contour.
                It defines also the maximum number of lines in the contour to find.
            start_line_index (int, optional): The starting line to find searching for
                the contour. Defaults to 0.

        Returns:
            (Contour): a Contour object
        """

        def display(lines):
            if debug:
                # image.show_image_and_lines(
                #     image=img,
                #     lines=lines,
                #     colors_lines=[
                #         (50 * i, 255 - 50 * i, 255) for i in range(len(_lines))
                #     ],
                # )
                pass

        if debug:
            assert img is not None
            img = img.copy()

        _lines = copy.deepcopy(np.array(lines))
        construct_contour = []
        is_contour_found = False
        idx_line_closest_point = start_line_index
        i = 0
        while not is_contour_found and i < max_iteration:
            cur_line = _lines[idx_line_closest_point]
            cur_geoline = Line(cur_line[0], cur_line[1])
            cur_point = cur_line[1]
            construct_contour.append(cur_line)
            _lines = np.delete(_lines, idx_line_closest_point, axis=0)

            if len(_lines) == 0:
                print("No more lines will do the same operation as no point detected")

            display(lines=_lines)

            # find the closest point to the current one and associated line
            lines2points = _lines.reshape(len(_lines) * 2, 2)
            distances_from_cur_point = np.linalg.norm(lines2points - cur_point, axis=1)
            # print(len(_lines), len(distances_from_cur_point), distances_from_cur_point)
            idx_closest_points = np.nonzero(
                distances_from_cur_point < min_dist_threshold
            )[0]

            if (
                len(idx_closest_points) > 1
            ):  # more than one point close to the current point
                # TODO
                # maybe just take the closest may be more complicated than that
                raise RuntimeError("More than one point close to the current point")
            elif len(idx_closest_points) == 0:
                first_line = construct_contour[0]
                first_point = first_line[0]
                distance_end_to_first_points = np.linalg.norm(first_point - cur_point)
                if distance_end_to_first_points < min_dist_threshold:
                    first_geoline = Line(first_line[0], first_line[1])
                    # TODO sometimes multiples intersection example 7
                    # print(cur_geoline.intersection(first_geoline)[0].evalf(n=7))
                    intersect_point = np.array(
                        cur_geoline.intersection(first_geoline)[0].evalf(n=7)
                    )
                    construct_contour[-1][1] = intersect_point
                    construct_contour[0][0] = intersect_point
                    is_contour_found = True
                    break
                else:
                    raise RuntimeError("No point detected close to the current point")

            idx_closest_point = int(idx_closest_points[0])
            idx_line_closest_point = int(np.floor(idx_closest_point / 2))

            # arrange the line so that the closest point is in the first place
            idx_point_in_line = 0 if (idx_closest_point / 2).is_integer() else 1
            line_closest_point = _lines[idx_line_closest_point]
            if idx_point_in_line == 1:  # flip points positions
                _lines[idx_line_closest_point] = np.flip(line_closest_point, axis=0)

            display(lines=[cur_line, line_closest_point])

            # find intersection point between the two lines
            geoline_closest_point = Line(line_closest_point[0], line_closest_point[1])
            intersect_point = np.array(
                cur_geoline.intersection(geoline_closest_point)[0].evalf(n=7)
            )

            # update values in arrays
            cur_line[1] = intersect_point
            line_closest_point[0] = intersect_point
            construct_contour[i][1] = intersect_point

            display(lines=[cur_line, line_closest_point])

            i += 1

        contour_lines = np.array(construct_contour)
        display(lines=[cur_line, line_closest_point])
        cnt = Contour.from_lines(contour_lines)
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

    @staticmethod
    def _reduce_noise(points: np.ndarray, max_dist_threshold: float = 5) -> np.ndarray:
        """Remove some points that describe an unwanted noise.

        Args:
            points (np.ndarray): array of shape (n, 2)
            max_dist_threshold (float, optional): minimum distance to suppress points.
                Defaults to 5.

        Returns:
            (np.ndarray): points without noisy and close points
        """
        # TODO
        return np.zeros(0)

    @staticmethod
    def _reduce_collinear(
        points: np.ndarray,
        margin_error_angle: float = DEFAULT_MARGIN_ANGLE_ERROR,
        n_iterations: int = 1,
    ) -> np.ndarray:
        """Remove close collinear points.
        Useful to clean a contour with a lot of points.

        Args:
            points (np.ndarray): array of shape (n, 2)
            margin_error_angle (float, optional): minimum distance to suppress points.
                Defaults to DEFAULT_MARGIN_ANGLE_ERROR.

        Returns:
            (np.ndarray): points without any collinear close points
        """
        for _ in range(n_iterations):
            idx_to_remove = []
            before_n_points = len(points)
            for i, cur_point in enumerate(points):
                if i == len(points) - 1:
                    i = -1  # so that next point is the first in that case
                elif i == len(points) - 2:
                    i = -2
                next_point = points[i + 1]
                next_next_point = points[i + 2]
                seg1 = Segment(points=[cur_point, next_point])
                seg2 = Segment(points=[next_point, next_next_point])
                if seg1.is_collinear(
                    segment=seg2, margin_error_angle=margin_error_angle
                ):
                    idx_to_remove.append(i + 1)

            points = np.delete(points, idx_to_remove, 0)

            # verify that the current iteration reduced the number of points
            after_n_points = len(points)
            if before_n_points == after_n_points:
                break

        return points

    @staticmethod
    def _reduce_by_distance(
        points: np.ndarray, min_dist_threshold: float = 2
    ) -> np.ndarray:
        # remove consecutive very close points
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
    def _reduce_by_distance_limit_n_successive_deletion(
        points: np.ndarray,
        min_dist_threshold: float = 2,
        limit_n_successive_deletion: int = 2,
        n_iterations: int = 1,
    ) -> np.ndarray:
        # remove consecutive very close points
        for _ in range(n_iterations):
            idx_to_remove: list[int] = []
            cur_idx_to_remove: list[int] = []
            before_n_points = len(points)
            for i, cur_point in enumerate(points):
                if i == len(points) - 1:
                    i = -1  # so that next point is the first in that case
                next_point = points[i + 1]
                distance = Segment(points=[cur_point, next_point]).length
                if distance < min_dist_threshold:
                    if len(cur_idx_to_remove) == 0 or (
                        len(cur_idx_to_remove) > 0 and i == cur_idx_to_remove[-1] + 1
                    ):
                        cur_idx_to_remove.append(i)
                    else:
                        cur_idx_to_remove = []
                else:
                    cur_idx_to_remove = []
                # print("Segment:", [cur_point.tolist(), next_point.tolist()], "distance", distance, idx_to_remove)
                if len(cur_idx_to_remove) >= limit_n_successive_deletion:
                    first_idx_pt, last_idx_pt = (
                        cur_idx_to_remove[0],
                        cur_idx_to_remove[-1],
                    )
                    centroid_point = Contour(
                        points=points[first_idx_pt:last_idx_pt]
                    ).centroid.astype(int)
                    points[first_idx_pt] = centroid_point
                    idx_to_remove = (
                        idx_to_remove + cur_idx_to_remove[1:]
                    )  # all except first point idx
                    cur_idx_to_remove = []

            points = np.delete(points, idx_to_remove, 0)

            # verify that the current iteration reduced the number of points
            after_n_points = len(points)
            if before_n_points == after_n_points:
                break

        return points

    @staticmethod
    def _reduce_by_triangle_area(
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

    @staticmethod
    def _reduce_by_weighted_orders_areas(
        points: np.ndarray,
        min_importance: float = 1,
        order: int = 1,
        weights: Optional[np.ndarray] = None,
        metric: str = "geo",
    ):
        """This an extension of the implementation of the Visvalingam–Whyatt algorithm.
        This algorithm is defined in the reduce by triangle areas method.

        The idea of this algorithm is to better detect the edges and keep these
        precious points. So we want to put a lot of importance on the edge but very
        low importance on the points very close to the optimum edge point.

        The resumed idea is: put a lot of importance on order-0 area and a lot less on the order-k
        areas for all k > 1.

        Args:
            points (np.ndarray): _description_
            min_triangle_area (float, optional): _description_. Defaults to 1.
            extension_order (int, optional): a
            weights (np.ndarray, optional):
            metric (str, optional):
        """
        if order == 0:
            return Contour._reduce_by_triangle_area(
                points=points, min_triangle_area=min_importance
            )
        n_points = len(points)
        n_areas = order + 1
        areas = np.zeros(shape=n_areas)
        if weights is None or len(weights) != n_areas:
            weights = np.full(shape=n_areas, fill_value=1 / n_areas)
        idx_to_remove = []
        for i in range(n_points):
            if i == n_points - 1 - n_areas:
                i = -1 - n_areas  # so that next point is the first in that case
            for k in range(n_areas):
                areas[k] = Contour(points=points[i - 1 - k : i + 2 + k]).area

            if metric == "geo":
                importance = np.prod(np.power(areas, weights))
            elif metric == "add":
                importance = (1 + areas[0]) ** weights[0] - np.sum(
                    np.dot(1 + areas[1:], weights[1:])
                )

            if importance < min_importance:
                idx_to_remove.append(i)

        reduced_points = np.delete(np.asarray(points), idx_to_remove, 0)

        return reduced_points

    # ------------------------------- CLASSIC METHODS ----------------------------------

    def is_auto_intersected(self) -> bool:
        """Whether any of the segments intersect another segment in the same set

        Returns:
            bool: True if at least two lines intersect, False otherwise
        """
        # TODO
        return False

    def add_point(self, point: np.ndarray, index: int):
        self.points = np.concatenate(
            [self.points[:index], [point], self.points[index:]]
        )
        return self

    def rearrange_first_point_is_at_index(self, index: int):
        self.points = np.concatenate([self.points[index:], self.points[:index]])
        return self

    def rearrange_first_point_closest_to_reference_point(
        self, reference_point: np.ndarray = np.zeros(shape=(2,))
    ):
        shifted_points = self.points - reference_point
        distances = np.linalg.norm(shifted_points, axis=1)
        idx_min_dist = np.argmin(distances).astype(int)
        return self.rearrange_first_point_is_at_index(index=idx_min_dist)

    # ------------------------------- Fundamental Methods ------------------------------

    def is_equal(self, contour: Contour, dist_margin_error: float = 5):
        if self.n_points != contour.n_points:
            # if the contours do not have the same number of points they can not be similar
            return False

        # check if each points composing the contours are close to each other
        new_cnt = contour.copy().rearrange_first_point_closest_to_reference_point(
            self.points[0]
        )
        points_diff = new_cnt.points - self.points
        distances = np.linalg.norm(points_diff, axis=1)
        max_distance = np.max(distances)
        return max_distance <= dist_margin_error
