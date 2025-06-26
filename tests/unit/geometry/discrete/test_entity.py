"""
Test GeometryEntity file
"""

import pytest
import numpy as np

from src.geometry import Polygon, Segment, Rectangle


class TestEntityBasics:
    def test_perimeter(self):
        rect = Rectangle([[0, 0], [0, 1], [1, 1], [1, 0]])
        assert rect.perimeter == 4

    def test_xmax(self):
        rect = Rectangle.unit()
        assert rect.xmax == 1

    def test_xmin(self):
        rect = Rectangle.unit()
        assert rect.xmin == 0

    def test_ymax(self):
        rect = Rectangle.unit()
        assert rect.ymax == 1

    def test_ymin(self):
        rect = Rectangle.unit()
        assert rect.ymin == 0

    def test_eq_true(self):
        rect = Rectangle([[0, 0], [0, 1], [1, 1], [1, 0]])
        rect2 = Rectangle([[0, 0], [0, 1], [1, 1], [1, 0]])
        assert rect == rect2

    def test_eq_false(self):
        rect = Rectangle([[0, 0], [0, 1], [1, 1], [1, 0]])
        rect2 = Rectangle([[0, 1], [0, 1], [1, 1], [1, 0]])
        assert not (rect == rect2)

    def test_eq_error(self):
        rect = Rectangle([[0, 0], [0, 1], [1, 1], [1, 0]])
        rect2 = "mystr"
        assert rect != rect2

    def test_add_float(self):
        rect = Rectangle([[0, 0], [0, 1], [1, 1], [1, 0]])
        rect = rect + 1
        assert np.array_equal(rect.asarray, np.array([[1, 1], [1, 2], [2, 2], [2, 1]]))

    def test_add_array(self):
        rect = Rectangle([[0, 0], [0, 1], [1, 1], [1, 0]])
        rect = rect + np.array([-1, 1])
        assert np.array_equal(
            rect.asarray, np.array([[-1, 1], [-1, 2], [0, 2], [0, 1]])
        )

    def test_sub_float(self):
        rect = Rectangle([[0, 0], [0, 1], [1, 1], [1, 0]])
        rect = rect - 1
        assert np.array_equal(
            rect.asarray, np.array([[-1, -1], [-1, 0], [0, 0], [0, -1]])
        )

    def test_sub_array(self):
        rect = Rectangle([[0, 0], [0, 1], [1, 1], [1, 0]])
        rect = rect - np.array([1, -1])
        assert np.array_equal(
            rect.asarray, np.array([[-1, 1], [-1, 2], [0, 2], [0, 1]])
        )

    def test_mul_float(self):
        rect = Rectangle([[0, 0], [0, 1], [1, 1], [1, 0]])
        rect = rect * 2
        assert np.array_equal(rect.asarray, np.array([[0, 0], [0, 2], [2, 2], [2, 0]]))

    def test_mul_array(self):
        rect = Rectangle([[0, 0], [0, 1], [1, 1], [1, 0]])
        rect = rect * np.array([2, -2])
        assert np.array_equal(
            rect.asarray, np.array([[0, 0], [0, -2], [2, -2], [2, 0]])
        )

    def test_div_float(self):
        rect = Rectangle([[0, 0], [0, 1], [1, 1], [1, 0]])
        rect = rect / 2.0
        assert np.array_equal(
            rect.asarray, np.array([[0, 0], [0, 1 / 2], [1 / 2, 1 / 2], [1 / 2, 0]])
        )

    def test_div_array(self):
        rect = Rectangle([[0, 0], [0, 1], [1, 1], [1, 0]], is_cast_int=False)
        rect = rect / np.array([2, -2])
        assert np.array_equal(
            rect.asarray, np.array([[0, 0], [0, -1 / 2], [1 / 2, -1 / 2], [1 / 2, 0]])
        )


class TestEntityShift:
    def test_shift_segment(self):
        seg = Segment([[1, 1], [2, 2]])
        shift_vector = [0, -2]
        assert np.array_equal(
            a1=seg.shift(vector=shift_vector).asarray, a2=np.array([[1, -1], [2, 0]])
        )

    def test_shift_segment_far_vector(self):
        seg = Segment([[1, 1], [2, 2]])
        shift_vector = [[10, 10], [10, 8]]
        assert np.array_equal(
            seg.shift(vector=shift_vector).asarray, np.array([[1, -1], [2, 0]])
        )

    def test_shift_segment_error(self):
        seg = Segment([[1, 1], [2, 2]])
        shift_vector = [[10, 10], [2, 2], [8, 9]]
        with pytest.raises(ValueError):
            seg.shift(vector=shift_vector)


class TestEntityRotate:
    def test_segment_rotate_pi_over_4(self):
        seg = Segment([[1, 1], [2, 2]])
        assert np.array_equal(
            np.round(seg.rotate(angle=np.pi / 4).asarray, 5),
            np.round([[1.5, 1.5 - np.sqrt(2) / 2], [1.5, 1.5 + np.sqrt(2) / 2]], 5),
        )

    def test_segment_rotate_pi_over_4_counter_clockwise_pivot(self):
        seg = Segment([[1, 1], [2, 2]])
        seg.rotate(angle=np.pi / 4, is_clockwise=False, pivot=[1, 1])
        assert np.array_equal(
            np.round(seg.asarray, 5),
            np.round([[1, 1], [1 + np.sqrt(2), 1]], 5),
        )

    def test_segment_rotate_pi_over_4_degree(self):
        seg = Segment([[1, 1], [2, 2]])
        assert np.array_equal(
            np.round(seg.rotate(angle=45, is_degree=True).asarray, 5),
            np.round([[1.5, 1.5 - np.sqrt(2) / 2], [1.5, 1.5 + np.sqrt(2) / 2]], 5),
        )

    def test_segment_rotate_pi_over_2(self):
        seg = Segment([[1, 1], [2, 2]])
        assert np.array_equal(
            seg.rotate(angle=np.pi / 2).asarray, np.array([[2, 1], [1, 2]])
        )

    def test_segment_rotate_pi(self):
        seg = Segment([[1, 1], [2, 2]])
        assert np.array_equal(
            np.round(seg.rotate(angle=np.pi).asarray, 5), np.round([[2, 2], [1, 1]], 5)
        )

    def test_segment_rotate_neg_pi_over_4(self):
        seg = Segment([[1, 1], [2, 2]])
        assert np.array_equal(
            np.round(seg.rotate(angle=-np.pi / 4).asarray, 5),
            np.round([[1.5 - np.sqrt(2) / 2, 1.5], [1.5 + np.sqrt(2) / 2, 1.5]], 5),
        )

    def test_segment_rotate_neg_pi_over_2(self):
        seg = Segment([[1, 1], [2, 2]])
        assert np.array_equal(
            seg.rotate(angle=-np.pi / 2).asarray, np.array([[1, 2], [2, 1]])
        )

    def test_segment_rotate_neg_pi(self):
        seg = Segment([[1, 1], [2, 2]])
        assert np.array_equal(
            np.round(seg.rotate(angle=-np.pi).asarray, 5), np.round([[2, 2], [1, 1]], 5)
        )

    def test_segment_rotate_around_image_center_pi(self):
        seg = Segment([[1, 1], [2, 2]])
        side = 100
        img = np.zeros(shape=(side, side))
        assert np.array_equal(
            seg.rotate_around_image_center(img=img, angle=np.pi).asarray,
            np.array([[side - 1, side - 1], [side - 2, side - 2]]),
        )

    def test_segment_rotate_around_image_center_pi_over_2(self):
        seg = Segment([[1, 1], [2, 2]])
        side = 100
        img = np.zeros(shape=(side, side))
        assert np.array_equal(
            seg.rotate_around_image_center(img=img, angle=np.pi / 2).asarray,
            np.array([[side - 1, 1], [side - 2, 2]]),
        )


class TestEntityEnclosing:
    def test_enclosing_axis_aligned_bbox(self):
        polygon = Polygon(
            points=[
                [10, 70],
                [30, 30],
                [30, 20],
                [50, 20],
                [60, 30],
                [70, 20],
                [80, 40],
                [70, 50],
                [50, 80],
                [40, 70],
            ]
        )
        expected_aabb = Polygon([[10, 20], [81, 20], [81, 81], [10, 81]])
        assert polygon.enclosing_axis_aligned_bbox().is_equal(expected_aabb)

    def test_enclosing_oriented_bbox(self):
        polygon = Polygon(
            points=[
                [10, 70],
                [30, 30],
                [30, 20],
                [50, 20],
                [60, 30],
                [70, 20],
                [80, 40],
                [70, 50],
                [50, 80],
                [40, 70],
            ]
        )
        expected_obb = Polygon([[9, 70], [35, 6], [85, 26], [60, 90]])
        assert polygon.enclosing_oriented_bbox().is_equal(expected_obb)

    def test_enclosing_convex_hull(self):
        polygon = Polygon(
            points=[
                [10, 70],
                [30, 30],
                [30, 20],
                [50, 20],
                [60, 30],
                [70, 20],
                [80, 40],
                [70, 50],
                [50, 80],
                [40, 70],
            ]
        )
        expected_aabb = Polygon([[10, 70], [30, 20], [70, 20], [80, 40], [50, 80]])
        assert polygon.enclosing_convex_hull().is_equal(expected_aabb)


class TestEntityIntersection:
    def test_intersection_no_point(self):
        seg = Segment([[0, 0], [10, 10]])
        seg1 = Segment([[0, 10], [0, 50]])
        intersection = seg.intersection(other=seg1)
        assert np.array_equal(intersection, np.array([]))

    def test_intersection_one_point(self):
        seg = Segment([[0, 0], [10, 10]])
        seg1 = Segment([[0, 10], [10, 0]])
        intersection = seg.intersection(other=seg1)
        assert np.array_equal(intersection, np.array([[5, 5]]))

    def test_intersection_two_points_vertical_line(self):
        seg = Segment([[0, 0], [0, 10]])
        rect = Rectangle([[-2, 2], [-2, 5], [2, 5], [2, 2]])
        intersection = seg.intersection(other=rect)
        assert np.array_equal(
            intersection, np.array([[0, 2], [0, 5]])
        ) or np.array_equal(intersection, np.array([[0, 5], [0, 2]]))

    def test_intersection_two_points_horizontal_line(self):
        seg = Segment([[-5, 4], [5, 4]])
        rect = Rectangle([[-2, 2], [-2, 6], [2, 6], [2, 2]])
        intersection = seg.intersection(other=rect)
        assert np.array_equal(
            intersection, np.array([[2, 4], [-2, 4]])
        ) or np.array_equal(intersection, np.array([[-2, 4], [2, 4]]))

    def test_intersection_two_points_diagonal_line(self):
        seg = Segment([[0, 0], [5, 5]])
        rect = Rectangle([[-2, 1], [-2, 4], [2, 4], [2, 1]])
        intersection = seg.intersection(other=rect)
        assert np.array_equal(
            intersection, np.array([[1, 1], [2, 2]])
        ) or np.array_equal(intersection, np.array([[2, 2], [1, 1]]))


class TestEntityContains:
    def test_rect_contained_equal_shapes(self):
        rect1 = Rectangle.unit()
        assert rect1.contains(rect1)

    def test_rect_contained_in_rect_no_common_border(self):
        rect1 = Rectangle.unit()
        rect2 = Rectangle.unit()
        rect1.asarray += 1
        rect2.asarray = rect2.asarray * 3
        assert rect2.contains(rect1)

    def test_rect_contained_in_rect_common_borders(self):
        rect1 = Rectangle.unit()
        rect2 = Rectangle.unit()
        rect2.asarray = rect2.asarray * 2
        assert rect2.contains(rect1)

    def test_rect_not_contained_without_intersection(self):
        rect1 = Rectangle.unit()
        rect2 = Rectangle.unit()
        rect2.asarray = rect2.asarray + 1
        rect1.asarray = rect1.asarray * (-1) - 1
        assert not rect2.contains(rect1)

    def test_rect_not_contained_with_intersection(self):
        rect1 = Rectangle.unit()
        rect2 = Rectangle.unit()
        rect2.asarray = rect2.asarray * 2
        rect1.asarray = rect1.asarray * 2 - 1
        assert not rect2.contains(rect1)


class TestEntityClamp:
    def test_clamp_within_bounds(self):
        rect = Rectangle([[0, 0], [0, 2], [2, 2], [2, 0]])
        rect.clamp(xmin=0, xmax=2, ymin=0, ymax=2)
        assert np.array_equal(rect.asarray, np.array([[0, 0], [0, 2], [2, 2], [2, 0]]))

    def test_clamp_exceeding_bounds(self):
        rect = Rectangle([[-1, -1], [0, 3], [3, 3], [3, -1]])
        rect.clamp(xmin=0, xmax=2, ymin=0, ymax=2)
        assert np.array_equal(rect.asarray, np.array([[0, 0], [0, 2], [2, 2], [2, 0]]))

    def test_clamp_negative_bounds(self):
        rect = Rectangle([[-3, -3], [-1, -1], [1, 1], [3, 3]])
        rect.clamp(xmin=-2, xmax=2, ymin=-2, ymax=2)
        assert np.array_equal(
            rect.asarray, np.array([[-2, -2], [-1, -1], [1, 1], [2, 2]])
        )

    def test_clamp_no_clipping(self):
        rect = Rectangle([[0, 0], [1, 1], [2, 2], [3, 3]])
        rect.clamp(xmin=-10, xmax=10, ymin=-10, ymax=10)
        assert np.array_equal(rect.asarray, np.array([[0, 0], [1, 1], [2, 2], [3, 3]]))

    def test_clamp_single_axis_clipping(self):
        rect = Rectangle([[0, 0], [1, 1], [2, 2], [3, 3]])
        rect.clamp(xmin=1, xmax=2)
        assert np.array_equal(rect.asarray, np.array([[1, 0], [1, 1], [2, 2], [2, 3]]))


class TestEntityNormalize:
    def test_normalize_positive_values(self):
        rect = Rectangle([[0, 0], [0, 2], [2, 2], [2, 0]])
        rect.normalize(x=2, y=2)
        assert np.array_equal(rect.asarray, np.array([[0, 0], [0, 1], [1, 1], [1, 0]]))

    def test_normalize_negative_values(self):
        rect = Rectangle([[0, 0], [0, -2], [-2, -2], [-2, 0]])
        rect.normalize(x=-2, y=-2)
        assert np.array_equal(rect.asarray, np.array([[0, 0], [0, 1], [1, 1], [1, 0]]))

    def test_normalize_mixed_values(self):
        rect = Rectangle([[0, 0], [0, 4], [2, 4], [2, 0]])
        rect.normalize(x=2, y=4)
        assert np.array_equal(rect.asarray, np.array([[0, 0], [0, 1], [1, 1], [1, 0]]))

    def test_normalize_with_zeros(self):
        rect = Rectangle([[0, 0], [0, 2], [2, 2], [2, 0]])
        with pytest.raises(ValueError):
            rect.normalize(x=0, y=2)

    def test_normalize_identity(self):
        rect = Rectangle([[0, 0], [0, 2], [2, 2], [2, 0]])
        rect.normalize(x=1, y=1)
        assert np.array_equal(rect.asarray, np.array([[0, 0], [0, 2], [2, 2], [2, 0]]))


class TestEntityDistVerticesToPoint:

    def test_shortest_dist_to_point_inside(self):
        rect = Rectangle([[0, 0], [0, 2], [2, 2], [2, 0]])
        point = np.array([1, 1])
        assert rect.shortest_dist_vertices_to_point(point) == np.sqrt(2)

    def test_shortest_dist_to_point_on_vertex(self):
        rect = Rectangle([[0, 0], [0, 2], [2, 2], [2, 0]])
        point = np.array([0, 0])
        assert rect.shortest_dist_vertices_to_point(point) == 0.0

    def test_shortest_dist_to_point_outside(self):
        rect = Rectangle([[0, 0], [0, 2], [2, 2], [2, 0]])
        point = np.array([3, 3])
        assert rect.shortest_dist_vertices_to_point(point) == np.sqrt(2)

    def test_shortest_dist_to_point_negative_coordinates(self):
        rect = Rectangle([[-2, -2], [-2, 0], [0, 0], [0, -2]])
        point = np.array([-1, -1])
        assert rect.shortest_dist_vertices_to_point(point) == np.sqrt(2)

    def test_shortest_dist_to_point_far_away(self):
        rect = Rectangle([[0, 0], [0, 2], [2, 2], [2, 0]])
        point = np.array([10, 10])
        assert rect.shortest_dist_vertices_to_point(point) == np.sqrt(2 * 8**2)

    def test_longest_dist_to_point_inside(self):
        rect = Rectangle([[0, 0], [0, 2], [2, 2], [2, 0]])
        point = np.array([1, 1])
        assert rect.longest_dist_vertices_to_point(point) == np.sqrt(2)

    def test_longest_dist_to_point_on_vertex(self):
        rect = Rectangle([[0, 0], [0, 2], [2, 2], [2, 0]])
        point = np.array([0, 0])
        assert rect.longest_dist_vertices_to_point(point) == 2 * np.sqrt(2)

    def test_longest_dist_to_point_outside(self):
        rect = Rectangle([[0, 0], [0, 2], [2, 2], [2, 0]])
        point = np.array([3, 3])
        assert rect.longest_dist_vertices_to_point(point) == np.sqrt(2 * 3**2)

    def test_longest_dist_to_point_negative_coordinates(self):
        rect = Rectangle([[-2, -2], [-2, 0], [0, 0], [0, -2]])
        point = np.array([-1, -1])
        assert rect.longest_dist_vertices_to_point(point) == np.sqrt(2)

    def test_longest_dist_to_point_far_away(self):
        rect = Rectangle([[0, 0], [0, 2], [2, 2], [2, 0]])
        point = np.array([10, 10])
        assert rect.longest_dist_vertices_to_point(point) == np.sqrt(2 * 10**2)


class TestEntitySharedApproxVertices:
    def test_shared_vertices_with_overlap(self):
        entity1 = Rectangle([[0, 0], [0, 2], [2, 2], [2, 0]])
        entity2 = Rectangle([[1, 1], [1, 3], [3, 3], [3, 1]])
        shared_vertices = entity1.find_shared_approx_vertices(
            entity2, margin_dist_error=1.5
        )
        expected_vertices = np.array([[0, 0], [0, 2], [2, 2], [2, 0]])
        assert np.array_equal(shared_vertices, expected_vertices)

    def test_shared_vertices_no_overlap(self):
        entity1 = Rectangle([[0, 0], [0, 2], [2, 2], [2, 0]])
        entity2 = Rectangle([[3, 3], [3, 5], [5, 5], [5, 3]])
        shared_vertices = entity1.find_shared_approx_vertices(
            entity2, margin_dist_error=1.5
        )
        expected_vertices = np.array([[2, 2]])
        assert np.array_equal(shared_vertices, expected_vertices)

    def test_shared_vertices_no_overlap_no_shared(self):
        entity1 = Rectangle([[0, 0], [0, 2], [2, 2], [2, 0]])
        entity2 = Rectangle([[3, 3], [3, 5], [5, 5], [5, 3]])
        shared_vertices = entity1.find_shared_approx_vertices(
            entity2, margin_dist_error=1.4
        )
        assert shared_vertices.size == 0

    def test_shared_vertices_exact_match(self):
        entity1 = Rectangle([[0, 0], [0, 2], [2, 2], [2, 0]])
        entity2 = Rectangle([[0, 0], [0, 2], [2, 2], [2, 0]])
        shared_vertices = entity1.find_shared_approx_vertices(
            entity2, margin_dist_error=0.001
        )
        expected_vertices = np.array([[0, 0], [0, 2], [2, 2], [2, 0]])
        assert np.array_equal(shared_vertices, expected_vertices)

    def test_shared_vertices_partial_overlap(self):
        entity1 = Rectangle([[0, 0], [0, 2], [2, 2], [2, 0]])
        entity2 = Rectangle([[2, 2], [2, 4], [4, 4], [4, 2]])
        shared_vertices = entity1.find_shared_approx_vertices(
            entity2, margin_dist_error=0.5
        )
        expected_vertices = np.array([[2, 2]])
        assert np.array_equal(shared_vertices, expected_vertices)

    def test_shared_vertices_with_large_margin(self):
        entity1 = Rectangle([[0, 0], [0, 2], [2, 2], [2, 0]])
        entity2 = Rectangle([[3, 3], [3, 5], [5, 5], [5, 3]])
        shared_vertices = entity1.find_shared_approx_vertices(
            entity2, margin_dist_error=5
        )
        expected_vertices = np.array([[0, 0], [0, 2], [2, 2], [2, 0]])
        assert np.array_equal(shared_vertices, expected_vertices)


class TestEntityVerticesFarFrom:

    def test_vertices_far_from_all_far(self):
        rect = Rectangle([[0, 0], [0, 2], [2, 2], [2, 0]])
        points = np.array([[10, 10]])
        far_vertices = rect.find_vertices_far_from(points=points, min_distance=5)
        assert np.array_equal(far_vertices, rect.asarray)

    def test_vertices_far_from_all_close(self):
        rect = Rectangle([[0, 0], [0, 2], [2, 2], [2, 0]])
        points = np.array([[1, 1]])
        far_vertices = rect.find_vertices_far_from(points=points, min_distance=2)
        assert far_vertices.size == 0

    def test_vertices_far_from_some_far(self):
        rect = Rectangle([[0, 0], [0, 2], [2, 2], [2, 0]])
        points = np.array([[0, 0], [2, 2]])
        far_vertices = rect.find_vertices_far_from(points=points, min_distance=1)
        expected_vertices = np.array([[0, 2], [2, 0]])
        assert np.array_equal(far_vertices, expected_vertices)

    def test_vertices_far_from_multiple_points(self):
        rect = Rectangle([[0, 0], [0, 2], [2, 2], [2, 0]])
        points = np.array([[0, 0], [1, 1], [2, 2]])
        far_vertices = rect.find_vertices_far_from(points=points, min_distance=1)
        expected_vertices = np.array([[0, 2], [2, 0]])
        assert np.array_equal(far_vertices, expected_vertices)

    def test_vertices_far_from_large_min_distance(self):
        rect = Rectangle([[0, 0], [0, 2], [2, 2], [2, 0]])
        points = np.array([[1, 1]])
        far_vertices = rect.find_vertices_far_from(points=points, min_distance=10)
        assert far_vertices.size == 0

    def test_vertices_far_from_zero_min_distance(self):
        rect = Rectangle([[0, 0], [0, 2], [2, 2], [2, 0]])
        points = np.array([[0.001, 0.001]])
        far_vertices = rect.find_vertices_far_from(points=points, min_distance=0)
        expected_vertices = np.array([[0, 0], [0, 2], [2, 2], [2, 0]])
        assert np.array_equal(far_vertices, expected_vertices)
