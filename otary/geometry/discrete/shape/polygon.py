"""
Polygon class to handle complexity with polygon calculation
"""

from __future__ import annotations

import copy
from typing import Optional, Self, Sequence, TYPE_CHECKING
import logging

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from shapely import LinearRing, Polygon as SPolygon

from otary.geometry.entity import GeometryEntity
from otary.geometry.discrete.linear.entity import LinearEntity
from otary.geometry.discrete.entity import DiscreteGeometryEntity
from otary.geometry import Segment, Vector, LinearSpline
from otary.geometry.utils.tools import get_shared_point_indices

if TYPE_CHECKING:
    from otary.geometry.discrete.shape.rectangle import Rectangle


class Polygon(DiscreteGeometryEntity):
    """Polygon class which defines a polygon object which means any closed-shape"""

    def __init__(self, points: NDArray | list, is_cast_int: bool = False) -> None:
        if len(points) <= 2:
            raise ValueError(
                "Cannot create a Polygon since it must have 3 or more points"
            )
        super().__init__(points=points, is_cast_int=is_cast_int)

    # ---------------------------------- OTHER CONSTRUCTORS ----------------------------

    @classmethod
    def from_lines(cls, lines: NDArray) -> Polygon:
        """The lines should describe a perfect closed shape polygon

        Args:
            lines (NDArray): array of lines of shape (n, 2, 2)

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
    def from_linear_entities_returns_vertices_ix(
        cls, linear_entities: Sequence[LinearEntity]
    ) -> tuple[Polygon, list[int]]:
        """Convert a list of linear entities to polygon.
        Beware: this method assumes entities are sorted and connected.

        Args:
            linear_entities (Sequence[LinearEntity]): List of linear entities.

        Returns:
            (Polygon, list[int]): polygon and indices of first vertex of each entity
        """
        points = []
        vertices_ix: list[int] = []
        current_ix = 0
        for linear_entity in linear_entities:
            if not isinstance(linear_entity, LinearEntity):
                raise TypeError(
                    f"Expected a list of LinearEntity, but got {type(linear_entity)}"
                )
            entity_points = linear_entity.points[:-1, :]
            points.append(entity_points)
            vertices_ix.append(current_ix)
            current_ix += len(entity_points)

        points = np.concatenate(points, axis=0)
        polygon = Polygon(points=points)
        return polygon, vertices_ix

    @classmethod
    def from_linear_entities(
        cls,
        linear_entities: Sequence[LinearEntity],
    ) -> Polygon:
        """Convert a list of linear entities to polygon.

        Beware: the method assumes entities are sorted and connected.

        Args:
            linear_entities (Sequence[LinearEntity]): List of linear entities.

        Returns:
            Polygon: polygon representation of the linear entity
        """
        return cls.from_linear_entities_returns_vertices_ix(linear_entities)[0]

    @classmethod
    def from_unordered_lines_approx(
        cls,
        lines: NDArray,
        max_dist_thresh: float = 50,
        max_iterations: int = 50,
        start_line_index: int = 0,
        img: Optional[NDArray] = None,
        is_debug_enabled: bool = False,
    ) -> Polygon:
        # pylint: disable=too-many-positional-arguments,too-many-arguments
        """Create a Polygon object from an unordered list of lines that approximate a
        closed-shape. They approximate in the sense that they do not necessarily
        share common points. This method computes the intersection points between lines.

        Args:
            img (_type_): array of shape (lx, ly)
            lines (NDArray): array of lines of shape (n, 2, 2)
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

        def debug_visualize(seg: NDArray):
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
    def centroid(self) -> NDArray:
        """Compute the centroid point which can be seen as the center of gravity
        or center of mass of the shape

        Returns:
            NDArray: centroid point
        """
        M = cv2.moments(self.asarray.astype(np.float32).reshape((-1, 1, 2)))

        # Avoid division by zero
        if M["m00"] != 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            centroid = np.asarray([cx, cy])
        else:
            centroid = self.center_mean

        return centroid

    @property
    def area(self) -> float:
        """Compute the area of the geometry entity

        Returns:
            float: area value
        """
        return cv2.contourArea(self.points.astype(np.int32))

    @property
    def perimeter(self) -> float:
        """Compute the perimeter of the geometry entity

        Returns:
            float: perimeter value
        """
        return cv2.arcLength(self.points.astype(np.float32), True)

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

    @property
    def edges(self) -> NDArray:
        """Get the lines that compose the geometry entity.

        Args:
            points (NDArray): array of points of shape (n, 2)

        Returns:
            NDArray: array of lines of shape (n, 2, 2)
        """
        return np.stack([self.points, np.roll(self.points, shift=-1, axis=0)], axis=1)

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
        idx_max_dist = self.find_vertice_ix_farthest_from(point=refpoint)
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

    def is_clockwise(self, is_y_axis_down: bool = False) -> bool:
        """Determine if a polygon points go clockwise using the Shoelace formula.

        True if polygon vertices order is clockwise in the "y-axis points up"
        referential.

        Args:
            is_y_axis_down (bool, optional): If is_y_axis_down is True, then the image
                referential is used where y axis points down.

        Returns:
            bool: True if clockwise, False if counter-clockwise
        """
        x = self.asarray[:, 0]
        y = self.asarray[:, 1]

        x_next = np.roll(x, -1)
        y_next = np.roll(y, -1)

        s = np.sum((x_next - x) * (y_next + y))

        is_clockwise = bool(s > 0)  # Clockwise if positive (OpenCV's convention)

        if is_y_axis_down:  # in referential where y axis points down
            return not is_clockwise

        return is_clockwise

    def as_linear_spline(self, index: int = 0) -> LinearSpline:
        """Get the polygon as a LinearSpline object.
        This simply means a LinearSpline object with the same points as the Polygon
        but with an extra point: the one at the index.

        Returns:
            LinearSpline: linear spline from polygon
        """
        if index < 0:
            index += len(self)

        index = index % len(self)

        return LinearSpline(
            points=np.concat(
                [self.asarray[index : len(self)], self.asarray[0 : index + 1]], axis=0
            )
        )

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
        if dilate_scale != 1:
            surface = self.copy().expand(scale=dilate_scale).shapely_surface
        else:
            surface = self.shapely_surface
        return surface.contains(other.shapely_surface)

    def score_vertices_in_points(self, points: NDArray, max_distance: float) -> NDArray:
        """Returns a score of 0 or 1 for each point in the polygon if it is close
        enough to any point in the input points.

        Args:
            points (NDArray): list of 2D points
            margin_dist_error (float): mininum distance to consider two points as
                close enough to be considered as the same points

        Returns:
            NDArray: a list of score for each point in the contour
        """

        indices = get_shared_point_indices(
            points_to_check=self.asarray,
            checkpoints=points,
            margin_dist_error=max_distance,
            method="close",
            cond="any",
        )
        score = np.bincount(indices, minlength=len(self))
        return score

    def find_vertices_between(self, start_index: int, end_index: int) -> NDArray:
        """Get the vertices between two indices.

        Returns always the vertices between start_index and end_index using the
        natural order of the vertices in the contour.

        By convention, if start_index == end_index, then it returns the whole contour
        plus the vertice at start_index.

        Args:
            start_index (int): index of the first vertex
            end_index (int): index of the last vertex

        Returns:
            NDArray: array of vertices
        """
        if start_index < 0:
            start_index += len(self)
        if end_index < 0:
            end_index += len(self)

        start_index = start_index % len(self)
        end_index = end_index % len(self)

        if start_index > end_index:
            vertices = np.concat(
                [
                    self.asarray[start_index : len(self)],
                    self.asarray[0 : end_index + 1],
                ],
                axis=0,
            )
        elif start_index == end_index:
            vertices = self.as_linear_spline(index=start_index).asarray
        else:
            vertices = self.asarray[start_index : end_index + 1]

        return vertices

    def find_interpolated_point_and_prev_ix(
        self, start_index: int, end_index: int, pct_dist: float
    ) -> tuple[NDArray, int]:
        """Return a point along the contour path from start_idx to end_idx (inclusive),
        at a relative distance pct_dist ∈ [0, 1] along that path.

        By convention, if start_index == end_index, then use the whole contour
        start at this index position.

        Parameters:
            start_idx (int): Index of the start point in the contour
            end_idx (int): Index of the end point in the contour
            pct_dist (float): Value in [0, 1], 0 returns start, 1 returns end.
                Any value in [0, 1] returns a point between start and end that is
                pct_dist along the path.

        Returns:
            NDArray: Interpolated point [x, y]
        """
        if not (0 <= pct_dist <= 1):
            raise ValueError("pct_dist must be in [0, 1]")

        if start_index < 0:
            start_index += len(self)
        if end_index < 0:
            end_index += len(self)

        start_index = start_index % len(self)
        end_index = end_index % len(self)

        path = LinearSpline(
            points=self.find_vertices_between(
                start_index=start_index, end_index=end_index
            )
        )

        point, index = path.find_interpolated_point_and_prev_ix(pct_dist=pct_dist)
        index = (index + start_index) % len(self)

        return point, index

    def find_interpolated_point(
        self, start_index: int, end_index: int, pct_dist: float
    ) -> NDArray:
        """Return a point along the contour path from start_idx to end_idx (inclusive),
        at a relative distance pct_dist ∈ [0, 1] along that path.

        By convention, if start_index == end_index, then use the whole contour
        start at this index position.

        Parameters:
            start_idx (int): Index of the start point in the contour
            end_idx (int): Index of the end point in the contour
            pct_dist (float): Value in [0, 1], 0 returns start, 1 returns end.
                Any value in [0, 1] returns a point between start and end that is
                pct_dist along the path.

        Returns:
            NDArray: Interpolated point [x, y]
        """
        return self.find_interpolated_point_and_prev_ix(
            start_index=start_index, end_index=end_index, pct_dist=pct_dist
        )[0]

    def normal_point(
        self,
        start_index: int,
        end_index: int,
        dist_along_edge_pct: float,
        dist_from_edge: float,
        is_outward: bool = True,
    ) -> NDArray:
        """Compute the outward normal point.
        This is a point that points toward the outside of the polygon

        Args:
            start_index (int): start index for the edge selection
            end_index (int): end index for the edge selection
            dist_along_edge_pct (float): distance along the edge to place the point
            dist_from_edge (float): distance outward from the edge
            is_outward (bool, optional): True if the normal points to the outside of
                the polygon. False if the normal points to the inside of the polygon. Defaults to True.

        Returns:
            NDArray: 2D point as array
        """
        assert 0.0 <= dist_along_edge_pct <= 1.0

        pt_interpolated, prev_ix = self.find_interpolated_point_and_prev_ix(
            start_index=start_index, end_index=end_index, pct_dist=dist_along_edge_pct
        )

        edge = Vector(
            points=[self.asarray[prev_ix], self.asarray[(prev_ix + 1) % len(self)]]
        )

        normal = edge.normal().normalized

        pt_plus = pt_interpolated + dist_from_edge * normal
        pt_minus = pt_interpolated - dist_from_edge * normal

        dist_plus = np.linalg.norm(pt_plus - self.centroid)
        dist_minus = np.linalg.norm(pt_minus - self.centroid)

        # choose the point which distance to the center is greater
        if dist_plus > dist_minus:
            if is_outward:
                return pt_plus
            return pt_minus

        if is_outward:
            return pt_minus
        return pt_plus

    def inter_area(self, other: Polygon) -> float:
        """Inter area with another Polygon

        Args:
            other (Polygon): other Polygon

        Returns:
            float: inter area value
        """
        inter_pts = cv2.intersectConvexConvex(self.asarray, other.asarray)
        if inter_pts[0] > 0:
            inter_area = cv2.contourArea(inter_pts[1])
        else:
            inter_area = 0.0
        return inter_area

    def union_area(self, other: Polygon) -> float:
        """Union area with another Polygon

        Args:
            other (Polygon): other Polygon

        Returns:
            float: union area value
        """
        return self.area + other.area - self.inter_area(other)

    def iou(self, other: Polygon) -> float:
        """Intersection over union with another Polygon

        Args:
            other (Polygon): other Polygon

        Returns:
            float: intersection over union value
        """
        inter_area = self.inter_area(other)

        # optimized not to compute twice the inter area
        union_area = self.area + other.area - inter_area

        if union_area == 0:
            return 0.0
        return inter_area / union_area

    # ---------------------------- MODIFICATION METHODS -------------------------------

    def add_vertice(self, point: NDArray, index: int) -> Self:
        """Add a point at a given index in the Polygon object

        Args:
            point (NDArray): point to be added
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

    def rearrange_first_vertice_at_index(self, index: int) -> Self:
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

    def rearrange_first_vertice_closest_to_point(
        self, point: NDArray = np.zeros(shape=(2,))
    ) -> Polygon:
        """Rearrange the list of vertices that defines the Polygon so that the first
        point in the list of vertices is the one that is the closest by distance to
        the reference point.

        Args:
            reference_point (NDArray): point that is taken as a reference in the
                space to find the one in the Polygon list of points that is the
                closest to this reference point. Default to origin point [0, 0].

        Returns:
            Polygon: Polygon which is the exact same one but with a rearranged list
                of points.
        """
        idx_min_dist = self.find_vertice_ix_closest_from(point=point)
        return self.rearrange_first_vertice_at_index(index=idx_min_dist)

    def reorder_clockwise(self, is_y_axis_down: bool = False) -> Polygon:
        """Reorder the vertices of the polygon in clockwise order where the first point
        stays the same.

        Args:
            is_y_axis_down (bool, optional): True if cv2 is used. Defaults to False.

        Returns:
            Polygon: reordered polygon
        """
        if self.is_clockwise(is_y_axis_down=is_y_axis_down):
            return self
        self.asarray = np.roll(self.asarray[::-1], shift=1, axis=0)
        return self

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
        if scale == 1.0:  # no rescaling
            return self

        center = self.centroid
        self.asarray = self.asarray.astype(float)
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

    def to_image_crop_referential(
        self,
        other: Polygon,
        crop: Rectangle,
        image_crop_shape: Optional[tuple[int, int]] = None,
    ) -> Polygon:
        """This function can be useful for a very specific need:
        In a single image you have two same polygons and their coordinates are defined
        in this image referential.

        You want to obtain the original polygon and all its vertices information
        in the image crop referential to match the other polygon within it.

        This method manipulates three referentials:
        1. image referential (main referential)
        2. crop referential
        3. image crop referential. It is different from the crop referential
            because the width and height of the crop referential may not be the same.

        Args:
            other (Polygon): other Polygon in the image referential
            crop_rect (Rectangle): crop rectangle in the image referential
            image_crop_shape (tuple[int, int], optionla): [width, height] of the crop
                image. If None, the shape is assumed to be directly the crop shape.


        Returns:
            Polygon: original polygon in the image crop referential
        """
        assert crop.contains(other=other)
        crop_width = int(crop.get_width_from_topleft(0))
        crop_height = int(crop.get_height_from_topleft(0))

        if image_crop_shape is None:
            image_crop_shape = (crop_width, crop_height)

        # self polygon in the original image shifted and normalized
        aabb_main = self.enclosing_axis_aligned_bbox()
        contour_main_shifted_normalized = self.copy().shift(
            vector=-np.asarray([self.xmin, self.ymin])
        ) / np.array(
            [aabb_main.get_width_from_topleft(0), aabb_main.get_height_from_topleft(0)]
        )

        # AABB of the polygon in the crop referential
        aabb_crop = other.enclosing_axis_aligned_bbox()
        aabb_crop_normalized = (
            aabb_crop - np.asarray([crop.xmin, crop.ymin])
        ) / np.array([crop_width, crop_height])

        # obtain the self polygon in the image crop referential
        aabb_crop2 = aabb_crop_normalized * np.array(image_crop_shape)
        new_polygon = contour_main_shifted_normalized * np.array(
            [
                aabb_crop2.get_width_from_topleft(0),
                aabb_crop2.get_height_from_topleft(0),
            ]
        ) + np.asarray([aabb_crop2.xmin, aabb_crop2.ymin])

        return new_polygon

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
        new_cnt = polygon.copy().rearrange_first_vertice_closest_to_point(
            self.points[0]
        )
        points_diff = new_cnt.points - self.points
        distances = np.linalg.norm(points_diff, axis=1)
        max_distance = np.max(distances)
        return max_distance <= dist_margin_error

    def __str__(self) -> str:
        return (
            self.__class__.__name__
            + "(start="
            + self.asarray[0].tolist().__str__()
            + ", end="
            + self.asarray[-1].tolist().__str__()
            + ")"
        )

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + "(start="
            + self.asarray[0].tolist().__str__()
            + ", end="
            + self.asarray[-1].tolist().__str__()
            + ")"
        )
