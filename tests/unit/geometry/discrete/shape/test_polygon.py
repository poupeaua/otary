"""
Unit tests for Polygon geometry class
"""

import numpy as np
import pytest

from otary.geometry import Polygon, Segment, Vector, LinearSpline
from otary.geometry.discrete.shape.rectangle import Rectangle


class TestPolygonInstantiationBasic:

    def test_polygon_instantiation_with_numpy_array(self):
        arr = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        polygon = Polygon(arr)
        assert np.array_equal(polygon.asarray, arr)

    def test_polygon_instantiation_with_list(self):
        points = [[0, 0], [1, 0], [1, 1], [0, 1]]
        polygon = Polygon(points)
        assert np.array_equal(polygon.asarray, np.array(points))

    def test_polygon_instantiation_cast_int(self):
        points = [[0.1, 0.9], [1.2, 0.2], [1.8, 1.7], [0.3, 1.1]]
        polygon = Polygon(points, is_cast_int=True)
        assert np.issubdtype(polygon.asarray.dtype, np.integer)

    def test_polygon_instantiation_too_few_points_2(self):
        with pytest.raises(ValueError):
            Polygon([[0, 0], [1, 1]])

    def test_polygon_instantiation_too_few_points_1(self):
        with pytest.raises(ValueError):
            Polygon([[0, 0]])

    def test_polygon_instantiation_too_few_points_0(self):
        with pytest.raises(ValueError):
            Polygon([])


class TestPolygonIsConvex:
    def test_is_convex_true(self):
        rect = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
        assert rect.is_convex

    def test_is_convex_false(self):
        polygon = Polygon(
            points=[
                [1, 7],
                [3, 3],
                [3, 2],
                [5, 2],
                [6, 3],
                [7, 2],
                [8, 4],
                [7, 5],
                [5, 8],
                [4, 7],
            ]
        )
        assert not polygon.is_convex


class TestPolygonIsEqual:
    def test_polygon_is_equal_same_exact_polygons(self):
        cnt1 = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
        cnt2 = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
        assert cnt1.is_equal(cnt2)

    def test_polygon_is_equal_very_close_polygons(self):
        cnt1 = Polygon([[0, 0], [100, 0], [100, 100], [0, 100]])
        cnt2 = Polygon([[1, -2], [99, -2], [103, 98], [-2, 101]])
        assert cnt1.is_equal(cnt2, dist_margin_error=5)

    def test_polygon_is_equal_false(self):
        cnt1 = Polygon([[0, 0], [100, 0], [100, 100], [0, 100]])
        cnt2 = Polygon([[1, -2], [95, -3], [103, 98], [-2, 101]])
        assert not cnt1.is_equal(cnt2, dist_margin_error=5)

    def test_polygons_different_n_points_is_equal_false(self):
        cnt1 = Polygon([[0, 0], [100, 0], [100, 100], [0, 100], [10, 5]])
        cnt2 = Polygon([[1, -2], [95, -3], [103, 98], [-2, 101]])
        assert not cnt1.is_equal(cnt2, dist_margin_error=5)


class TestPolygonIsRegular:
    def test_polygon_is_regular_square(self):
        cnt1 = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
        assert cnt1.is_regular(margin_dist_error_pct=0.0001)

    def test_polygon_is_regular_rectangle(self):
        cnt1 = Polygon([[0, 0], [10, 0], [10, 100], [0, 100]])
        assert cnt1.is_regular(margin_dist_error_pct=0.0001)

    def test_polygon_is_regular_almost_perfect(self):
        cnt1 = Polygon([[0, 0], [100, 0], [100, 101.2], [0, 100]])
        assert cnt1.is_regular(margin_dist_error_pct=0.01)

    def test_polygon_is_not_regular_almost_perfect(self):
        cnt1 = Polygon([[0, 0], [100, 0], [100, 101.2], [0, 100]])
        assert not cnt1.is_regular(margin_dist_error_pct=0.001)

    def test_polygon_is_not_regular(self):
        cnt1 = Polygon([[0, 0], [100, 0], [101, 100], [0, 100]])
        assert not cnt1.is_regular(margin_dist_error_pct=0)

    @pytest.mark.parametrize("alpha", (2, 3, 5, 25, 50))
    def test_polygon_is_not_regular_hard_case(self, alpha: float):
        height_big_rect = 100
        height_small_rect = 10
        cnt1 = Polygon(
            [
                [0, 0],
                [alpha * height_big_rect, 0],
                [alpha * height_big_rect, height_big_rect],
                [alpha * height_small_rect, height_big_rect + height_small_rect],
            ]
        )
        assert not cnt1.is_regular(margin_dist_error_pct=0.01)

    def test_polygon_is_not_regular_losange(self):
        cnt1 = Polygon([[0, 0], [10, 100], [200, 0], [-10, 100]])
        assert not cnt1.is_regular(margin_dist_error_pct=0.01)

    def test_polygon_is_not_regular_trapeze_equal_length_diagonals(self):
        cnt1 = Polygon([[0, 0], [100, 10], [100, 50], [0, 60]])
        assert not cnt1.is_regular(margin_dist_error_pct=0.01)

    def test_polygon_is_not_regular_more_than_four_points(self):
        cnt1 = Polygon([[0, 0], [1, 0], [1, 1], [0, 0.5], [0.5, 0.5]])
        assert not cnt1.is_regular(margin_dist_error_pct=10)

    def test_polygon_is_not_regular_less_than_four_points(self):
        cnt1 = Polygon([[0, 0], [1, 0], [1.5, 1]])
        assert not cnt1.is_regular(margin_dist_error_pct=10)

    def test_is_regular_rectangle_on_single_line(self):
        """Important test that checks the multiple intersection points case"""
        # All points are colinear (degenerate rectangle)
        points = [[0, 0], [4, 0], [4, 0], [0, 0]]
        polygon = Polygon(points)
        # Should not be regular: diagonals do not intersect at a single point
        assert not polygon.is_regular(margin_dist_error_pct=0.01)


class TestPolygonSelfIntersect:
    def test_polygon_is_self_intersect(self):
        cnt1 = Polygon([[0, 0], [1, 0], [0, 1], [1, 1]])
        assert cnt1.is_self_intersected

    def test_polygon_is_not_self_intersect(self):
        cnt1 = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
        assert not cnt1.is_self_intersected


class TestPolygonClassMethods:
    def test_construct_from_lines(self):
        lines = np.array([[[0, 0], [2, 2]], [[2, 2], [5, 5]], [[5, 5], [0, 0]]])
        cnt = Polygon.from_lines(lines=lines)
        assert np.array_equal(cnt.edges, lines)

    def test_construct_from_lines_fails(self):
        lines = [[[0, 0], [2, 2]], [[2, 3], [5, 5]], [[5, 5], [0, 0]]]
        with pytest.raises(ValueError):
            Polygon.from_lines(lines=lines)

    def test_construct_from_lines_fails_last(self):
        lines = [[[0, 0], [2, 2]], [[2, 2], [5, 5]], [[5, 5], [0, 1]]]
        with pytest.raises(ValueError):
            Polygon.from_lines(lines=lines)


class TestPolygonAddPoint:
    def test_add_point(self):
        cnt = Polygon([[0, 0], [1, 1], [1, 0]])
        pt = [0, 1]
        assert np.array_equal(
            cnt.add_vertice(point=pt, index=1).asarray, [[0, 0], [0, 1], [1, 1], [1, 0]]
        )

    def test_add_point_last(self):
        cnt = Polygon([[0, 0], [1, 1], [1, 0]])
        pt = [0, 1]
        assert np.array_equal(
            cnt.add_vertice(point=pt, index=-1).asarray,
            [[0, 0], [1, 1], [1, 0], [0, 1]],
        )

    def test_add_point_first_neg(self):
        cnt = Polygon([[0, 0], [1, 1], [1, 0]])
        pt = [0, 1]
        assert np.array_equal(
            cnt.add_vertice(point=pt, index=-4).asarray,
            [[0, 1], [0, 0], [1, 1], [1, 0]],
        )

    def test_add_point_first(self):
        cnt = Polygon([[0, 0], [1, 1], [1, 0]])
        pt = [0, 1]
        assert np.array_equal(
            cnt.add_vertice(point=pt, index=0).asarray, [[0, 1], [0, 0], [1, 1], [1, 0]]
        )

    def test_add_point_index_too_big(self):
        cnt = Polygon([[0, 0], [1, 1], [1, 0]])
        pt = [0, 1]
        with pytest.raises(ValueError):
            cnt.add_vertice(point=pt, index=4)

    def test_add_point_index_too_small(self):
        cnt = Polygon([[0, 0], [1, 1], [1, 0]])
        pt = [0, 1]
        with pytest.raises(ValueError):
            cnt.add_vertice(point=pt, index=-5)


class TestPolygonRearrange:
    def test_rearrange_first_point_at_index_pos(self):
        cnt = Polygon([[0, 0], [1, 1], [1, 0]])
        assert np.array_equal(
            cnt.rearrange_first_vertice_at_index(index=1).asarray,
            [[1, 1], [1, 0], [0, 0]],
        )

    def test_rearrange_first_point_at_index_neg(self):
        cnt = Polygon([[0, 0], [1, 1], [1, 0]])
        assert np.array_equal(
            cnt.rearrange_first_vertice_at_index(index=-2).asarray,
            [[1, 1], [1, 0], [0, 0]],
        )

    def test_rearrange_first_point_at_index_too_big(self):
        cnt = Polygon([[0, 0], [1, 1], [1, 0]])
        with pytest.raises(ValueError):
            cnt.rearrange_first_vertice_at_index(index=3)

    def test_rearrange_first_point_at_index_too_small(self):
        cnt = Polygon([[0, 0], [1, 1], [1, 0]])
        with pytest.raises(ValueError):
            cnt.rearrange_first_vertice_at_index(index=-4)


class TestPolygonFromUnorderedLinesApprox:
    @pytest.mark.parametrize(
        "input,output",
        (
            (
                [  # square
                    [[0, 0], [0, 100]],
                    [[0, 101], [102, 98]],
                    [[101, 96], [102, 0]],
                    [[99, -1], [-1, 1]],
                ],
                [
                    [[0, 0], [0, 101]],
                    [[0, 101], [100, 98]],
                    [[100, 98], [102, -1]],
                    [[102, -1], [0, 0]],
                ],
            ),
            (
                [  # triangle
                    [[0, 0], [50, 100]],
                    [[50, 101], [102, 0]],
                    [[101, 0], [-1, 0]],
                ],
                [[[0, 0], [50, 100]], [[50, 100], [102, 0]], [[102, 0], [0, 0]]],
            ),
        ),
    )
    def test_cnt_fula_general(self, input: list, output: list):
        cnt = Polygon.from_unordered_lines_approx(input)
        assert np.all([o in output for o in cnt.edges.tolist()])

    @pytest.mark.parametrize(
        "input",
        (
            (
                [  # square
                    [[0, 0], [0, 100]],
                    [[0, 115], [102, 98]],
                    [[101, 96], [102, 0]],
                    [[99, -1], [-1, 1]],
                ]
            ),
            (
                [  # triangle
                    [[0, 0], [50, 100]],
                    [[50, 101], [102, 0]],
                    [[101, 0], [12, 0]],
                ]
            ),
        ),
    )
    def test_cnt_fula_dist_sup_thresh(self, input: list):
        with pytest.raises(RuntimeError):
            Polygon.from_unordered_lines_approx(input, max_dist_thresh=10)


class TestPolygonScoreEdgesInPoints:

    def test_score_edges_in_points_all_close(self):
        cnt = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
        points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        scores = cnt.score_vertices_in_points(points=points, max_distance=0.1)
        assert np.array_equal(scores, [1, 1, 1, 1])

    def test_score_edges_in_points_some_close(self):
        cnt = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
        points = np.array([[0, 0], [1, 0]])
        scores = cnt.score_vertices_in_points(points=points, max_distance=0.1)
        assert np.array_equal(scores, [1, 1, 0, 0])

    def test_score_edges_in_points_none_close(self):
        cnt = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
        points = np.array([[2, 2], [3, 3]])
        scores = cnt.score_vertices_in_points(points=points, max_distance=0.1)
        assert np.array_equal(scores, [0, 0, 0, 0])

    def test_score_edges_in_points_with_margin(self):
        cnt = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
        points = np.array([[0.05, 0.05], [1.05, 0.05]])
        scores = cnt.score_vertices_in_points(points=points, max_distance=0.1)
        assert np.array_equal(scores, [1, 1, 0, 0])

    def test_score_edges_in_points_duplicate_points(self):
        cnt = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
        points = np.array([[0, 0], [0, 0], [1, 0]])
        scores = cnt.score_vertices_in_points(points=points, max_distance=0.1)
        assert np.array_equal(scores, [1, 1, 0, 0])


class TestPolygonContains:
    def test_contains_true(self):
        polygon = Polygon([[0, 0], [4, 0], [4, 4], [0, 4]])
        other = Polygon([[1, 1], [3, 1], [3, 3], [1, 3]])
        assert polygon.contains(other)

    def test_contains_false(self):
        polygon = Polygon([[0, 0], [4, 0], [4, 4], [0, 4]])
        other = Polygon([[3, 3], [5, 3], [5, 5], [3, 5]])
        assert not polygon.contains(other)

    def test_contains_with_dilate_scale_true(self):
        polygon = Polygon([[0, 0], [4, 0], [4, 4], [0, 4]])
        other = Polygon([[3.5, 3.5], [4.5, 3.5], [4.5, 4.5], [3.5, 4.5]])
        assert polygon.contains(other, dilate_scale=1.5)

    def test_contains_with_dilate_scale_true2(self):
        polygon = Polygon([[0, 0], [4, 0], [4, 4], [0, 4]])
        other = Polygon([[5, 5], [6, 5], [6, 6], [5, 6]])
        assert polygon.contains(other, dilate_scale=2)

    def test_contains_with_dilate_scale_false(self):
        polygon = Polygon([[0, 0], [4, 0], [4, 4], [0, 4]])
        other = Polygon([[5, 5], [6, 5], [6, 6], [5, 6]])
        assert not polygon.contains(other, dilate_scale=1.9)

    def test_contains_invalid_dilate_scale(self):
        polygon = Polygon([[0, 0], [4, 0], [4, 4], [0, 4]])
        other = Polygon([[1, 1], [3, 1], [3, 3], [1, 3]])
        with pytest.raises(ValueError):
            polygon.contains(other, dilate_scale=0.5)


class TestPolygonExpand:
    def test_expand_valid_scale(self):
        polygon = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
        expanded_polygon = polygon.expand(scale=2)
        expected_points = [[-0.5, -0.5], [1.5, -0.5], [1.5, 1.5], [-0.5, 1.5]]
        assert np.allclose(expanded_polygon.asarray, expected_points)

    def test_expand_invalid_scale_less_than_one(self):
        polygon = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
        with pytest.raises(ValueError):
            polygon.expand(scale=0.5)

    def test_expand_no_change_scale_one(self):
        polygon = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
        expanded_polygon = polygon.expand(scale=1)
        assert np.array_equal(expanded_polygon.asarray, polygon.asarray)


class TestPolygonShrink:
    def test_shrink_valid_scale(self):
        polygon = Polygon([[0, 0], [2, 0], [2, 2], [0, 2]])
        shrunk_polygon = polygon.shrink(scale=2)
        expected_points = [[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5]]
        assert np.allclose(shrunk_polygon.asarray, expected_points)

    def test_shrink_invalid_scale_less_than_one(self):
        polygon = Polygon([[0, 0], [2, 0], [2, 2], [0, 2]])
        with pytest.raises(ValueError):
            polygon.shrink(scale=0.5)

    def test_shrink_no_change_scale_one(self):
        polygon = Polygon([[0, 0], [2, 0], [2, 2], [0, 2]])
        shrunk_polygon = polygon.shrink(scale=1)
        assert np.array_equal(shrunk_polygon.asarray, polygon.asarray)


class TestPolygonLengths:
    def test_lengths_square(self):
        polygon = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
        expected_lengths = [1, 1, 1, 1]
        assert np.allclose(polygon.lengths, expected_lengths)

    def test_lengths_rectangle(self):
        polygon = Polygon([[0, 0], [2, 0], [2, 1], [0, 1]])
        expected_lengths = [2, 1, 2, 1]
        assert np.allclose(polygon.lengths, expected_lengths)

    def test_lengths_triangle(self):
        polygon = Polygon([[0, 0], [3, 0], [1.5, 2]])
        expected_lengths = [
            3,  # base
            np.sqrt(2.25 + 4),  # left side
            np.sqrt(2.25 + 4),  # right side
        ]
        assert np.allclose(polygon.lengths, expected_lengths)

    def test_lengths_irregular_polygon(self):
        polygon = Polygon([[0, 0], [2, 0], [3, 1], [1, 2]])
        expected_lengths = [
            2,  # bottom
            np.sqrt(1 + 1),  # right diagonal
            np.sqrt(4 + 1),  # top diagonal
            np.sqrt(1 + 4),  # left diagonal
        ]
        assert np.allclose(polygon.lengths, expected_lengths)


class TestPolygonIsClockwise:
    def test_is_clockwise_false(self):
        polygon = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
        assert polygon.is_clockwise() is False

    def test_is_clockwise_true(self):
        polygon = Polygon([[0, 0], [0, 1], [1, 1], [1, 0]])
        assert polygon.is_clockwise() is True

    def test_is_clockwise_true_cv2(self):
        polygon = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
        assert polygon.is_clockwise(is_y_axis_down=True) is True

    def test_is_clockwise_false_cv2(self):
        polygon = Polygon([[0, 0], [0, 1], [1, 1], [1, 0]])
        assert polygon.is_clockwise(is_y_axis_down=True) is False

    def test_is_clockwise_triangle_false(self):
        polygon = Polygon([[0, 0], [1, 0], [0.5, 1]])
        assert polygon.is_clockwise() is False

    def test_is_clockwise_triangle_true(self):
        polygon = Polygon([[0, 0], [0.5, 1], [1, 0]])
        assert polygon.is_clockwise() is True


class TestPolygonAsLinearSpline:
    def test_as_linear_spline_default_index(self):
        polygon = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
        spline = polygon.as_linear_spline()
        expected_points = [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]
        assert np.array_equal(spline.points, expected_points)

    def test_as_linear_spline_positive_index(self):
        polygon = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
        spline = polygon.as_linear_spline(index=2)
        expected_points = [[1, 1], [0, 1], [0, 0], [1, 0], [1, 1]]
        assert np.array_equal(spline.points, expected_points)

    def test_as_linear_spline_negative_index(self):
        polygon = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
        spline = polygon.as_linear_spline(index=-2)
        expected_points = [[1, 1], [0, 1], [0, 0], [1, 0], [1, 1]]
        assert np.array_equal(spline.points, expected_points)

    def test_as_linear_spline_index_out_of_bounds(self):
        polygon = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
        spline = polygon.as_linear_spline(index=5)
        expected_points = [[1, 0], [1, 1], [0, 1], [0, 0], [1, 0]]
        assert np.array_equal(spline.points, expected_points)


class TestPolygonVerticesBetween:
    def test_vertices_between_positive_indices(self):
        polygon = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
        result = polygon.find_vertices_between(1, 3)
        expected = np.array([[1, 0], [1, 1], [0, 1]])
        assert np.array_equal(result, expected)

    def test_vertices_between_negative_indices(self):
        polygon = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
        result = polygon.find_vertices_between(-3, -1)
        expected = np.array([[1, 0], [1, 1], [0, 1]])
        assert np.array_equal(result, expected)

    def test_vertices_between_wraparound_indices(self):
        polygon = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
        result = polygon.find_vertices_between(3, 1)
        expected = np.array([[0, 1], [0, 0], [1, 0]])
        assert np.array_equal(result, expected)

    def test_vertices_between_same_start_end_index(self):
        polygon = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
        result = polygon.find_vertices_between(2, 2)
        expected = np.array([[1, 1], [0, 1], [0, 0], [1, 0], [1, 1]])
        assert np.array_equal(result, expected)

    def test_vertices_between_full_cycle(self):
        polygon = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
        result = polygon.find_vertices_between(0, 3)
        expected = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        assert np.array_equal(result, expected)


class TestPolygonInterpolatedPointAlongPolygon:
    def test_interpolated_point_start(self):
        polygon = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
        result = polygon.find_interpolated_point(0, 2, 0)
        expected = np.array([0, 0])
        assert np.array_equal(result, expected)

    def test_interpolated_point_end(self):
        polygon = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
        result = polygon.find_interpolated_point(0, 2, 1)
        expected = np.array([1, 1])
        assert np.array_equal(result, expected)

    def test_interpolated_point_middle(self):
        polygon = Polygon([[0, 0], [2, 0], [2, 2], [0, 2]])
        result = polygon.find_interpolated_point(0, 2, 0.5)
        expected = np.array([2, 0])
        assert np.array_equal(result, expected)

    def test_interpolated_point_negative_indices(self):
        polygon = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
        result = polygon.find_interpolated_point(-4, -2, 0.5)
        expected = np.array([1, 0])
        assert np.array_equal(result, expected)

    def test_interpolated_point_wraparound(self):
        polygon = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
        result = polygon.find_interpolated_point(3, 1, 0.5)
        expected = np.array([0, 0])
        assert np.array_equal(result, expected)

    def test_interpolated_point_invalid_pct_dist(self):
        polygon = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
        with pytest.raises(ValueError):
            polygon.find_interpolated_point(0, 2, -0.1)

    def test_interpolated_point_start_end_same(self):
        polygon = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
        result = polygon.find_interpolated_point(1, 1, 0.5)
        expected = np.array([0, 1])
        assert np.array_equal(result, expected)

    def test_interpolated_point_hard_case1(self):
        polygon = Polygon(
            points=np.array([[0, 0], [100, 100], [200, 100], [300, 200], [300, 0]])
        )
        result = polygon.find_interpolated_point(0, 3, 0.5)
        expected = np.array([150, 100])
        assert np.array_equal(result, expected)

    def test_interpolated_point_hard_case2(self):
        polygon = Polygon(
            points=np.array([[0, 0], [100, 100], [200, 100], [300, 200], [300, 0]])
        )
        result = polygon.find_interpolated_point(
            start_index=-2, end_index=0, pct_dist=0.4
        )
        expected = np.array([300, 0])
        assert np.array_equal(result, expected)

    def test_interpolated_point_hard_case3(self):
        polygon = Polygon(
            points=np.array([[0, 0], [100, 100], [200, 100], [300, 200], [300, 0]])
        )
        result = polygon.find_interpolated_point(
            start_index=1, end_index=-4, pct_dist=0.5
        )
        expected = np.array([300, 0])
        assert np.array_equal(result, expected)


class TestPolygonReorderClockwise:
    def test_reorder_clockwise_already_clockwise(self):
        polygon = Polygon([[0, 0], [0, 1], [1, 1], [1, 0]])
        reordered_polygon = polygon.reorder_clockwise()
        expected_points = [[0, 0], [0, 1], [1, 1], [1, 0]]
        assert np.array_equal(reordered_polygon.asarray, expected_points)

    def test_reorder_clockwise_not_clockwise(self):
        polygon = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
        reordered_polygon = polygon.reorder_clockwise()
        expected_points = [[0, 0], [0, 1], [1, 1], [1, 0]]
        assert np.array_equal(reordered_polygon.asarray, expected_points)

    def test_reorder_clockwise_already_clockwise_y_axis_down(self):
        polygon = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
        reordered_polygon = polygon.reorder_clockwise(is_y_axis_down=True)
        expected_points = [[0, 0], [1, 0], [1, 1], [0, 1]]
        assert np.array_equal(reordered_polygon.asarray, expected_points)

    def test_reorder_clockwise_not_clockwise_y_axis_down(self):
        polygon = Polygon([[0, 0], [0, 1], [1, 1], [1, 0]])
        reordered_polygon = polygon.reorder_clockwise(is_y_axis_down=True)
        expected_points = [[0, 0], [1, 0], [1, 1], [0, 1]]
        assert np.array_equal(reordered_polygon.asarray, expected_points)

    def test_reorder_clockwise_triangle(self):
        polygon = Polygon([[0, 0], [1, 0], [0.5, 1]])
        reordered_polygon = polygon.reorder_clockwise()
        expected_points = [[0, 0], [0.5, 1], [1, 0]]
        assert np.array_equal(reordered_polygon.asarray, expected_points)


class TestPolygonArea:
    def test_area_square(self):
        polygon = Polygon([[0, 0], [2, 0], [2, 2], [0, 2]])
        assert polygon.area == 4.0

    def test_area_rectangle(self):
        polygon = Polygon([[0, 0], [3, 0], [3, 2], [0, 2]])
        assert polygon.area == 6.0

    def test_area_triangle(self):
        polygon = Polygon([[0, 0], [4, 0], [0, 3]])
        assert polygon.area == 6.0

    def test_area_irregular_quadrilateral(self):
        polygon = Polygon([[0, 0], [4, 0], [3, 2], [0, 3]])
        # Area can be calculated using Shoelace formula: 0.5*|0*0+4*2+3*3+0*0 - (0*4+0*3+2*0+3*0)| = 0.5*|0+8+9+0 - (0+0+0+0)| = 0.5*17 = 8.5
        assert polygon.area == 8.5

    def test_area_negative_coordinates(self):
        polygon = Polygon([[-1, -1], [-1, 1], [1, 1], [1, -1]])
        assert polygon.area == 4.0

    def test_area_non_integer_coordinates(self):
        polygon = Polygon([[0.5, 0.5], [2.5, 0.5], [2.5, 2.5], [0.5, 2.5]])
        # cv2.contourArea casts to int, so all points become [0,0],[2,0],[2,2],[0,2], area=4
        assert polygon.area == 4.0


class TestPolygonPerimeter:
    def test_perimeter_square(self):
        polygon = Polygon([[0, 0], [2, 0], [2, 2], [0, 2]])
        # 4 sides of length 2
        assert polygon.perimeter == 8.0

    def test_perimeter_rectangle(self):
        polygon = Polygon([[0, 0], [3, 0], [3, 2], [0, 2]])
        # 2 sides of 3, 2 sides of 2
        assert polygon.perimeter == 10.0

    def test_perimeter_triangle(self):
        polygon = Polygon([[0, 0], [4, 0], [0, 3]])
        # Sides: 4, 5, 3
        assert polygon.perimeter == 12.0

    def test_perimeter_irregular_quadrilateral(self):
        polygon = Polygon([[0, 0], [4, 0], [3, 2], [0, 3]])
        # Calculate each side
        pts = np.array([[0, 0], [4, 0], [3, 2], [0, 3]])
        lengths = [
            np.linalg.norm(pts[0] - pts[1]),
            np.linalg.norm(pts[1] - pts[2]),
            np.linalg.norm(pts[2] - pts[3]),
            np.linalg.norm(pts[3] - pts[0]),
        ]
        expected = sum(lengths)
        assert np.isclose(polygon.perimeter, expected)

    def test_perimeter_negative_coordinates(self):
        polygon = Polygon([[-1, -1], [-1, 1], [1, 1], [1, -1]])
        # 4 sides of length 2
        assert polygon.perimeter == 8.0

    def test_perimeter_non_integer_coordinates(self):
        polygon = Polygon([[0.5, 0.5], [200.5, 0.5], [200.5, 200.5], [0.5, 200.5]])
        # Each side is 2, so perimeter is 8
        print(polygon.perimeter)
        assert np.isclose(polygon.perimeter, 800.0)


class TestPolygonNormalPoint:
    def test_normal_point_outward_square_horizontal_edge(self):
        polygon = Polygon([[0, 0], [2, 0], [2, 2], [0, 2]])
        # bottom edge from (0,0) to (2,0), midpoint is (1,0)
        # normal should point outward (down), so (1,0) + [0,-1] = (1,-1) for dist=1
        pt = polygon.normal_point(0, 1, 0.5, 1, is_outward=True)
        assert np.allclose(pt, [1, -1])

    def test_normal_point_inward_square_horizontal_edge(self):
        polygon = Polygon([[0, 0], [2, 0], [2, 2], [0, 2]])
        pt = polygon.normal_point(0, 1, 0.5, 1, is_outward=False)
        # inward should point up, so (1,0) + [0,1] = (1,1)
        assert np.allclose(pt, [1, 1])

    def test_normal_point_outward_square_vertical_edge(self):
        polygon = Polygon([[0, 0], [2, 0], [2, 2], [0, 2]])
        # right edge from (2,0) to (2,2), midpoint is (2,1)
        pt = polygon.normal_point(1, 2, 0.5, 1, is_outward=True)
        # outward should point right, so (2,1) + [1,0] = (3,1)
        assert np.allclose(pt, [3, 1])

    def test_normal_point_inward_square_vertical_edge(self):
        polygon = Polygon([[0, 0], [2, 0], [2, 2], [0, 2]])
        pt = polygon.normal_point(1, 2, 0.5, 1, is_outward=False)
        # inward should point left, so (2,1) + [-1,0] = (1,1)
        assert np.allclose(pt, [1, 1])


class TestPolygonFromLinearEntities:
    def test_from_linear_entities_returns_vertices_ix_segments(self):
        segs = [
            Segment([[0, 0], [1, 0]]),
            Segment([[1, 0], [1, 1]]),
            Segment([[1, 1], [0, 1]]),
            Segment([[0, 1], [0, 0]]),
        ]
        polygon, vertices_ix = Polygon.from_linear_entities_returns_vertices_ix(segs)
        expected_points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        assert np.array_equal(polygon.asarray, expected_points)
        assert vertices_ix == [0, 1, 2, 3]

    def test_from_linear_entities_returns_vertices_ix_linear_splines(self):
        # Multiple LinearSplines, each a segment of the polygon
        splines = [
            LinearSpline([[0, 0], [1, 0], [2, 0]]),
            LinearSpline([[2, 0], [2, 1], [2, 2]]),
            LinearSpline([[2, 2], [1, 2], [0, 2]]),
            LinearSpline([[0, 2], [0, 1], [0, 0]]),
        ]
        polygon, vertices_ix = Polygon.from_linear_entities_returns_vertices_ix(splines)
        expected_points = np.array(
            [[0, 0], [1, 0], [2, 0], [2, 1], [2, 2], [1, 2], [0, 2], [0, 1]]
        )
        assert np.array_equal(polygon.asarray, expected_points)
        assert vertices_ix == [0, 2, 4, 6]

    def test_from_linear_entities_returns_vertices_ix_mixed(self):
        # Mix of Segment, Vector, and LinearSpline
        seg = Segment([[0, 0], [2, 0]])
        v = Vector([[2, 0], [2, 2]])
        spline = LinearSpline([[2, 2], [1, 2], [0, 2]])
        v2 = Vector([[0, 2], [0, 0]])
        polygon, vertices_ix = Polygon.from_linear_entities_returns_vertices_ix(
            [seg, v, spline, v2]
        )
        expected_points = np.array([[0, 0], [2, 0], [2, 2], [1, 2], [0, 2]])
        assert np.array_equal(polygon.asarray, expected_points)
        assert vertices_ix == [0, 1, 2, 4]

    def test_from_linear_entities_returns_vertices_ix_empty(self):
        # Should raise ValueError if linear_entities is empty
        with pytest.raises(ValueError):
            Polygon.from_linear_entities_returns_vertices_ix([])

    def test_from_linear_entites_returns_vertices_ix_not_linear_entities(self):
        # Should raise TypeError if linear_entities is not a list of LinearEntities
        with pytest.raises(TypeError):
            Polygon.from_linear_entities_returns_vertices_ix([1, "str", 3.14])

    def test_from_linear_entities_returns_vertices_ix_not_connected(self):
        # Should raise ValueError if entities are not connected
        segs = [
            Segment([[0, 0], [1, 0]]),
            Segment([[2, 0], [2, 1]]),  # Not connected to previous
            Segment([[2, 1], [0, 1]]),
            Segment([[0, 1], [0, 0]]),
        ]
        with pytest.raises(ValueError):
            Polygon.from_linear_entities_returns_vertices_ix(segs)


class TestPolygonFromLinearEntitiesMethod:

    def test_from_linear_entities_mixed_types(self):
        seg = Segment([[0, 0], [2, 0]])
        v = Vector([[2, 0], [2, 2]])
        spline = LinearSpline([[2, 2], [1, 2], [0, 2]])
        v2 = Vector([[0, 2], [0, 0]])
        polygon = Polygon.from_linear_entities([seg, v, spline, v2])
        expected_points = np.array([[0, 0], [2, 0], [2, 2], [1, 2], [0, 2]])
        assert np.array_equal(polygon.asarray, expected_points)


class TestPolygonInterArea:
    def test_inter_area_identical_polygons(self):
        polygon1 = Polygon([[0, 0], [2, 0], [2, 2], [0, 2]])
        polygon2 = Polygon([[0, 0], [2, 0], [2, 2], [0, 2]])
        assert np.isclose(polygon1.inter_area(polygon2), polygon1.area)

    def test_inter_area_partial_overlap(self):
        polygon1 = Polygon([[0, 0], [2, 0], [2, 2], [0, 2]])
        polygon2 = Polygon([[1, 1], [3, 1], [3, 3], [1, 3]])
        # Overlapping area is a square from (1,1) to (2,2), area = 1
        assert np.isclose(polygon1.inter_area(polygon2), 1.0)

    def test_inter_area_no_overlap(self):
        polygon1 = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
        polygon2 = Polygon([[2, 2], [3, 2], [3, 3], [2, 3]])
        assert polygon1.inter_area(polygon2) == 0.0

    def test_inter_area_one_inside_another(self):
        polygon1 = Polygon([[0, 0], [4, 0], [4, 4], [0, 4]])
        polygon2 = Polygon([[1, 1], [3, 1], [3, 3], [1, 3]])
        # polygon2 is fully inside polygon1, so intersection is area of polygon2
        assert np.isclose(polygon1.inter_area(polygon2), polygon2.area)

    def test_inter_area_touching_edges(self):
        polygon1 = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
        polygon2 = Polygon([[1, 0], [2, 0], [2, 1], [1, 1]])
        # Touch at edge, intersection area should be 0
        assert polygon1.inter_area(polygon2) == 0.0

    def test_inter_area_touching_at_point(self):
        polygon1 = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
        polygon2 = Polygon([[1, 1], [2, 1], [2, 2], [1, 2]])
        # Touch at a single point, intersection area should be 0
        assert polygon1.inter_area(polygon2) == 0.0


class TestPolygonUnionArea:
    def test_union_area_identical_polygons(self):
        polygon1 = Polygon([[0, 0], [2, 0], [2, 2], [0, 2]])
        polygon2 = Polygon([[0, 0], [2, 0], [2, 2], [0, 2]])
        # Union area should be the same as the area of one polygon
        assert np.isclose(polygon1.union_area(polygon2), polygon1.area)

    def test_union_area_partial_overlap(self):
        polygon1 = Polygon([[0, 0], [2, 0], [2, 2], [0, 2]])
        polygon2 = Polygon([[1, 1], [3, 1], [3, 3], [1, 3]])
        # Each area is 4, overlap is 1, so union = 4 + 4 - 1 = 7
        assert np.isclose(polygon1.union_area(polygon2), 7.0)

    def test_union_area_no_overlap(self):
        polygon1 = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
        polygon2 = Polygon([[2, 2], [3, 2], [3, 3], [2, 3]])
        # No overlap, so union = 1 + 1 - 0 = 2
        assert np.isclose(polygon1.union_area(polygon2), 2.0)

    def test_union_area_one_inside_another(self):
        polygon1 = Polygon([[0, 0], [4, 0], [4, 4], [0, 4]])
        polygon2 = Polygon([[1, 1], [3, 1], [3, 3], [1, 3]])
        # polygon2 is fully inside polygon1, so union = area of polygon1
        assert np.isclose(polygon1.union_area(polygon2), polygon1.area)

    def test_union_area_touching_edges(self):
        polygon1 = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
        polygon2 = Polygon([[1, 0], [2, 0], [2, 1], [1, 1]])
        # Touch at edge, no overlap, union = 1 + 1 - 0 = 2
        assert np.isclose(polygon1.union_area(polygon2), 2.0)

    def test_union_area_touching_at_point(self):
        polygon1 = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
        polygon2 = Polygon([[1, 1], [2, 1], [2, 2], [1, 2]])
        # Touch at a single point, no overlap, union = 1 + 1 - 0 = 2
        assert np.isclose(polygon1.union_area(polygon2), 2.0)


class TestPolygonIOU:
    def test_iou_identical_polygons(self):
        polygon1 = Polygon([[0, 0], [2, 0], [2, 2], [0, 2]])
        polygon2 = Polygon([[0, 0], [2, 0], [2, 2], [0, 2]])
        assert np.isclose(polygon1.iou(polygon2), 1.0)

    def test_iou_partial_overlap(self):
        polygon1 = Polygon([[0, 0], [2, 0], [2, 2], [0, 2]])
        polygon2 = Polygon([[1, 1], [3, 1], [3, 3], [1, 3]])
        # Overlap area = 1, union area = 7
        assert np.isclose(polygon1.iou(polygon2), 1.0 / 7.0)

    def test_iou_no_overlap(self):
        polygon1 = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
        polygon2 = Polygon([[2, 2], [3, 2], [3, 3], [2, 3]])
        assert polygon1.iou(polygon2) == 0.0

    def test_iou_one_inside_another(self):
        polygon1 = Polygon([[0, 0], [4, 0], [4, 4], [0, 4]])
        polygon2 = Polygon([[1, 1], [3, 1], [3, 3], [1, 3]])
        # IOU = area of inner / area of outer
        assert np.isclose(polygon1.iou(polygon2), polygon2.area / polygon1.area)

    def test_iou_touching_edges(self):
        polygon1 = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
        polygon2 = Polygon([[1, 0], [2, 0], [2, 1], [1, 1]])
        assert polygon1.iou(polygon2) == 0.0

    def test_iou_touching_at_point(self):
        polygon1 = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
        polygon2 = Polygon([[1, 1], [2, 1], [2, 2], [1, 2]])
        assert polygon1.iou(polygon2) == 0.0

    def test_iou_union_area_zero(self):
        # Degenerate case: both polygons have zero area (all points the same)
        polygon1 = Polygon([[0, 0], [0, 0], [0, 0]])
        polygon2 = Polygon([[0, 0], [0, 0], [0, 0]])
        assert polygon1.iou(polygon2) == 0.0


class TestPolygonNormalPointMethod:
    def test_normal_point_outward_horizontal_edge(self):
        polygon = Polygon([[0, 0], [2, 0], [2, 2], [0, 2]])
        # bottom edge from (0,0) to (2,0), midpoint is (1,0)
        pt = polygon.normal_point(0, 1, 0.5, 1, is_outward=True)
        # outward normal should point down, so (1,0) + [0,-1] = (1,-1)
        assert np.allclose(pt, [1, -1])

    def test_normal_point_inward_horizontal_edge(self):
        polygon = Polygon([[0, 0], [2, 0], [2, 2], [0, 2]])
        pt = polygon.normal_point(0, 1, 0.5, 1, is_outward=False)
        # inward normal should point up, so (1,0) + [0,1] = (1,1)
        assert np.allclose(pt, [1, 1])

    def test_normal_point_outward_vertical_edge(self):
        polygon = Polygon([[0, 0], [2, 0], [2, 2], [0, 2]])
        pt = polygon.normal_point(1, 2, 0.5, 1, is_outward=True)
        # right edge from (2,0) to (2,2), midpoint is (2,1)
        # outward normal should point right, so (2,1) + [1,0] = (3,1)
        assert np.allclose(pt, [3, 1])

    def test_normal_point_inward_vertical_edge(self):
        polygon = Polygon([[0, 0], [2, 0], [2, 2], [0, 2]])
        pt = polygon.normal_point(1, 2, 0.5, 1, is_outward=False)
        # inward normal should point left, so (2,1) + [-1,0] = (1,1)
        assert np.allclose(pt, [1, 1])

    def test_normal_point_start_of_edge(self):
        polygon = Polygon([[0, 0], [2, 0], [2, 2], [0, 2]])
        with pytest.raises(ValueError):
            polygon.normal_point(0, 1, 0.0, 1, is_outward=True)

    def test_normal_point_end_of_edge(self):
        polygon = Polygon([[0, 0], [2, 0], [2, 2], [0, 2]])
        with pytest.raises(ValueError):
            polygon.normal_point(0, 1, 1.0, 1, is_outward=True)

    def test_normal_point_on_edge_special_case(self):
        polygon = Polygon([[0, 0], [2, 0], [2, 2], [0, 2]])
        with pytest.raises(ValueError):
            polygon.normal_point(0, 2, 0.5, 1, is_outward=True)

    def test_normal_point_invalid_pct(self):
        polygon = Polygon([[0, 0], [2, 0], [2, 2], [0, 2]])
        with pytest.raises(ValueError):
            polygon.normal_point(0, 1, -0.1, 1)
        with pytest.raises(ValueError):
            polygon.normal_point(0, 1, 1.1, 1)

    def test_normal_point_triangle(self):
        polygon = Polygon([[0, 0], [2, 0], [1, 2]])
        pt = polygon.normal_point(0, 1, 0.5, 1, is_outward=True)
        # For triangle, bottom edge midpoint is (1,0), normal points down
        assert np.allclose(pt, [1, -1])

    def test_normal_point_triangle_hard_case(self):
        polygon = Polygon([[0, 0], [2, 0], [1, 2]])
        pt = polygon.normal_point(1, 2, 0.5, np.sqrt(5), is_outward=True)
        # For triangle, bottom edge midpoint is (1,0), normal points down
        assert np.allclose(pt, [3.5, 2])

    def test_normal_point_non_unit_distance(self):
        polygon = Polygon([[0, 0], [2, 0], [2, 2], [0, 2]])
        pt = polygon.normal_point(0, 1, 0.5, 2, is_outward=True)
        # Should be (1,0) + 2*[0,-1] = (1,-2)
        assert np.allclose(pt, [1, -2])

    def test_normal_point_other_direction_polygon_outward(self):
        polygon = Polygon([[0, 0], [0, 2], [2, 2], [2, 0]])
        pt = polygon.normal_point(0, 1, 0.5, 1, is_outward=True)
        assert np.allclose(pt, [-1, 1])

    def test_normal_point_other_direction_polygon_inward(self):
        polygon = Polygon([[0, 0], [0, 2], [2, 2], [2, 0]])
        pt = polygon.normal_point(0, 1, 0.5, 1, is_outward=False)
        assert np.allclose(pt, [1, 1])


class TestPolygonStrRepr:
    def test_str_square(self):
        polygon = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
        s = str(polygon)
        assert s.startswith("Polygon(")
        assert "start=[0, 0]" in s
        assert "end=[0, 1]" in s

    def test_repr_square(self):
        polygon = Polygon([[0, 0], [1, 0], [1, 1], [0, 1]])
        r = repr(polygon)
        assert r.startswith("Polygon(")
        assert "start=[0, 0]" in r
        assert "end=[0, 1]" in r
        

class TestPolygonToImageCropReferential:
        
    def test_to_image_crop_referential_basic(self):
        # Main polygon in image referential
        poly_main = Polygon([[10, 10], [20, 10], [20, 20], [10, 20]])
        # Other polygon (same shape, different position)
        poly_other = Polygon([[2, 2], [8, 2], [8, 8], [2, 8]])
        # Crop rectangle in image referential
        crop = Rectangle.from_topleft_bottomright(topleft=[0, 0], bottomright=[10, 10], is_cast_int=True)
        # image_crop_shape is None (should default to crop size)
        result = poly_main.to_image_crop_referential(poly_other, crop)
        assert isinstance(result, Polygon)
        # The result should be a polygon (Polygon) with 4 points
        assert result.asarray.shape == (4, 2)

    def test_to_image_crop_referential_with_image_crop_shape(self):
        poly_main = Polygon([[10, 10], [20, 10], [20, 20], [10, 20]])
        poly_other = Polygon([[2, 2], [8, 2], [8, 8], [2, 8]])
        crop = Rectangle.from_topleft_bottomright(topleft=[0, 0], bottomright=[10, 10], is_cast_int=True)
        image_crop_shape = (100, 200)
        result = poly_main.to_image_crop_referential(poly_other, crop, image_crop_shape=image_crop_shape)
        assert isinstance(result, Polygon)
        assert result.asarray.shape == (4, 2)

    def test_to_image_crop_referential_assertion_fails(self):
        poly_main = Polygon([[10, 10], [20, 10], [20, 20], [10, 20]])
        poly_other = Polygon([[2, 2], [8, 2], [8, 8], [2, 8]])
        crop = Rectangle.from_topleft_bottomright(topleft=[0, 0], bottomright=[5, 5], is_cast_int=True)
        with pytest.raises(ValueError):
            poly_main.to_image_crop_referential(poly_other, crop)
