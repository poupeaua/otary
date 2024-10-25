"""
Unit tests for Contour geometry class
"""

import numpy as np
import pytest

from src.geometry import Contour


class TestContourIsEqual:
    def test_contour_is_equal_same_exact_contours(self):
        cnt1 = Contour([[0, 0], [1, 0], [1, 1], [0, 1]])
        cnt2 = Contour([[0, 0], [1, 0], [1, 1], [0, 1]])
        assert cnt1.is_equal(cnt2)

    def test_contour_is_equal_very_close_contours(self):
        cnt1 = Contour([[0, 0], [100, 0], [100, 100], [0, 100]])
        cnt2 = Contour([[1, -2], [99, -2], [103, 98], [-2, 101]])
        assert cnt1.is_equal(cnt2, dist_margin_error=5)

    def test_contour_is_equal_false(self):
        cnt1 = Contour([[0, 0], [100, 0], [100, 100], [0, 100]])
        cnt2 = Contour([[1, -2], [95, -3], [103, 98], [-2, 101]])
        assert not cnt1.is_equal(cnt2, dist_margin_error=5)

    def test_contours_different_n_points_is_equal_false(self):
        cnt1 = Contour([[0, 0], [100, 0], [100, 100], [0, 100], [10, 5]])
        cnt2 = Contour([[1, -2], [95, -3], [103, 98], [-2, 101]])
        assert not cnt1.is_equal(cnt2, dist_margin_error=5)


class TestContourIsregular:
    def test_contour_is_regular_square(self):
        cnt1 = Contour([[0, 0], [1, 0], [1, 1], [0, 1]])
        assert cnt1.is_regular(margin_dist_error_pct=0.0001)

    def test_contour_is_regular_rectangle(self):
        cnt1 = Contour([[0, 0], [10, 0], [10, 100], [0, 100]])
        assert cnt1.is_regular(margin_dist_error_pct=0.0001)

    def test_contour_is_regular_almost_perfect(self):
        cnt1 = Contour([[0, 0], [100, 0], [100, 101.2], [0, 100]])
        assert cnt1.is_regular(margin_dist_error_pct=0.01)

    def test_contour_is_not_regular_almost_perfect(self):
        cnt1 = Contour([[0, 0], [100, 0], [100, 101.2], [0, 100]])
        assert not cnt1.is_regular(margin_dist_error_pct=0.001)

    def test_contour_is_not_regular(self):
        cnt1 = Contour([[0, 0], [100, 0], [101, 100], [0, 100]])
        assert not cnt1.is_regular(margin_dist_error_pct=0)

    @pytest.mark.parametrize("alpha", (2, 3, 5, 25, 50))
    def test_contour_is_not_regular_hard_case(self, alpha: float):
        height_big_rect = 100
        height_small_rect = 10
        cnt1 = Contour(
            [
                [0, 0],
                [alpha * height_big_rect, 0],
                [alpha * height_big_rect, height_big_rect],
                [alpha * height_small_rect, height_big_rect + height_small_rect],
            ]
        )
        assert not cnt1.is_regular(margin_dist_error_pct=0.01)

    def test_contour_is_not_regular_losange(self):
        cnt1 = Contour([[0, 0], [10, 100], [200, 0], [-10, 100]])
        assert not cnt1.is_regular(margin_dist_error_pct=0.01)

    def test_contour_is_not_regular_trapeze_equal_length_diagonals(self):
        cnt1 = Contour([[0, 0], [100, 10], [100, 50], [0, 60]])
        assert not cnt1.is_regular(margin_dist_error_pct=0.01)

    def test_contour_is_not_regular_more_than_four_points(self):
        cnt1 = Contour([[0, 0], [1, 0], [1, 1], [0, 0.5], [0.5, 0.5]])
        assert not cnt1.is_regular(margin_dist_error_pct=10)

    def test_contour_is_not_regular_less_than_four_points(self):
        cnt1 = Contour([[0, 0], [1, 0], [1.5, 1]])
        assert not cnt1.is_regular(margin_dist_error_pct=10)


class TestContourSelfIntersect:
    def test_contour_is_self_intersect(self):
        cnt1 = Contour([[0, 0], [1, 0], [0, 1], [1, 1]])
        assert cnt1.is_self_intersected

    def test_contour_is_not_self_intersect(self):
        cnt1 = Contour([[0, 0], [1, 0], [1, 1], [0, 1]])
        assert not cnt1.is_self_intersected


class TestContourReduceMethods:
    @pytest.mark.parametrize(
        "input,output",
        (
            (
                [[0, 0], [100, 100], [500, 500], [0, 250]],
                [[0, 0], [500, 500], [0, 250]],
            ),
            (
                [[0, 0], [100, 100], [200, 200], [350, 350], [500, 500], [850, 120]],
                [[0, 0], [500, 500], [850, 120]],
            ),
            (
                [[0, 0], [100, 100], [200, 325], [350, 350], [500, 500], [120, 120]],
                [[100, 100], [200, 325], [350, 350]],
            ),
            (
                [
                    [0, 0],
                    [0, 100],
                    [0, 200],
                    [100, 200],
                    [250, 200],
                    [250, 120],
                    [250, 0],
                    [125, 0],
                ],
                [[0, 0], [0, 200], [250, 200], [250, 0]],
            ),
            # reduce collinear with perturbation
            (
                [[0, -1], [100, 101], [505, 498], [-2, 253]],
                [[0, -1], [505, 498], [-2, 253]],
            ),
            (
                [[0, 0], [102, 95], [202, 195], [345, 350], [501, 500], [850, 121]],
                [[0, 0], [501, 500], [850, 121]],
            ),
            (
                [[2, -1], [102, 100], [200, 326], [351, 350], [500, 502], [120, 121]],
                [[102, 100], [200, 326], [351, 350]],
            ),
            (
                [
                    [-1, 0],
                    [0, 100],
                    [0, 199],
                    [100, 200],
                    [253, 200],
                    [253, 120],
                    [248, -3],
                    [125, 0],
                ],
                [[-1, 0], [0, 199], [253, 200], [248, -3]],
            ),
        ),
    )
    def test_contour_reduce_collinear(self, input, output):
        points = Contour.reduce_collinear(points=input, n_iterations=3)
        assert np.array_equal(points, output)

    @pytest.mark.parametrize(
        "input,output",
        (
            (
                [[0, 0], [100, 100], [500, 500], [0, 250], [0, 2]],
                [[0, 0], [100, 100], [500, 500], [0, 250]],
            ),
            (
                [[0, 1], [100, 100], [500, 500], [0, 250], [0, 2], [1, 2], [2, 2]],
                [[0, 1], [100, 100], [500, 500], [0, 250]],
            ),
            (
                [[0, 0], [100, 100], [200, 200], [202, 199], [500, 500], [850, 120]],
                [[0, 0], [100, 100], [202, 199], [500, 500], [850, 120]],
            ),
            (
                [[0, 0], [100, 100], [200, 325], [350, 350], [351, 348], [-1, 1]],
                [[0, 0], [100, 100], [200, 325], [351, 348]],
            ),
            (
                [
                    [0, 0],
                    [-2, 199],
                    [0, 200],
                    [249, 200],
                    [250, 200],
                    [250, 200],
                    [250, -2],
                    [250, 0],
                ],
                [[0, 0], [0, 200], [250, 200], [250, 0]],
            ),
        ),
    )
    def test_contour_reduce_by_distance(self, input, output):
        points = Contour.reduce_by_distance(points=input, min_dist_threshold=5)
        assert np.array_equal(points, output)

    @pytest.mark.parametrize(
        "input,output",
        (
            (
                [[0, 0], [100, 100], [500, 500], [0, 250], [0, 2], [0, 5], [1, 5]],
                [[0, 0], [100, 100], [500, 500], [0, 250], [0, 2], [1, 5]],
            ),
            (
                [
                    [0, 0],
                    [100, 100],
                    [200, 200],
                    [202, 199],
                    [203, 198],
                    [500, 500],
                    [850, 120],
                ],
                [[0, 0], [100, 100], [200, 200], [203, 198], [500, 500], [850, 120]],
            ),
            (
                [[0, 0], [100, 100], [200, 325], [351, 348], [-1, 1], [1, 0], [1, 1]],
                [[0, 0], [100, 100], [200, 325], [351, 348], [-1, 1], [1, 1]],
            ),
            (
                [
                    [0, 0],
                    [-2, 199],
                    [0, 200],
                    [249, 200],
                    [250, 200],
                    [250, 201],
                    [250, -2],
                    [250, 0],
                ],
                [
                    [0, 0],
                    [-2, 199],
                    [0, 200],
                    [249, 200],
                    [250, 201],
                    [250, -2],
                    [250, 0],
                ],
            ),
        ),
    )
    def test_contour_reduce_by_distance_unsuccessive(self, input, output):
        points = Contour.reduce_by_distance_unsuccessive(
            points=input, min_dist_threshold=5, n_iterations=3
        )
        assert np.array_equal(points, output)

    @pytest.mark.parametrize(
        "input,output",
        (
            (
                [[0, 0], [100, 100], [500, 500], [0, 250], [0, 2], [0, 4], [2, 0]],
                [[0, 0], [100, 100], [500, 500], [0, 250], [0, 3], [2, 0]],
            ),
            (
                [[0, 0], [100, 100], [200, 325], [351, 348], [0, 0], [2, 0], [4, 2]],
                [[0, 0], [100, 100], [200, 325], [351, 348], [1, 0], [4, 2]],
            ),
        ),
    )
    def test_contour_reduce_by_distance_unsuccessive_mode_mean(self, input, output):
        points = Contour.reduce_by_distance_unsuccessive(
            points=input, min_dist_threshold=5, mode="mean"
        )
        assert np.array_equal(points, output)

    def test_contour_reduce_by_distance_unsuccessive_invalid_mode(self):
        with pytest.raises(ValueError):
            Contour.reduce_by_distance_unsuccessive(
                points=[], min_dist_threshold=5, mode="no-existing-mode"
            )

    @pytest.mark.parametrize(
        "input,output",
        (
            (
                [[0, 0], [100, 100], [500, 500], [0, 250], [0, 10], [0, 5], [1, 2]],
                [[0, 0], [500, 500], [0, 250]],
            ),
            (
                [
                    [0, 0],
                    [100, 100],
                    [200, 325],
                    [351, 348],
                    [0, 350],
                    [0, 200],
                    [0, 50],
                ],
                [[0, 0], [100, 100], [200, 325], [351, 348], [0, 350]],
            ),
            (
                [
                    [0, 0],
                    [0, 100],
                    [0, 200],
                    [100, 200],
                    [200, 200],
                    [200, 100],
                    [200, 0],
                    [100, 0],
                ],
                [
                    [0, 0],
                    [0, 200],
                    [200, 200],
                    [200, 0],
                ],
            ),
        ),
    )
    def test_contour_reduce_by_triangle_area(self, input, output):
        points = Contour.reduce_by_triangle_area(points=input, min_triangle_area=50)
        assert np.array_equal(points, output)


class TestContourClassMethods:
    def test_construct_from_lines(self):
        lines = np.array([[[0, 0], [2, 2]], [[2, 2], [5, 5]], [[5, 5], [0, 0]]])
        cnt = Contour.from_lines(lines=lines)
        assert np.array_equal(cnt.lines, lines)

    def test_construct_from_lines_fails(self):
        lines = [[[0, 0], [2, 2]], [[2, 3], [5, 5]], [[5, 5], [0, 0]]]
        with pytest.raises(ValueError):
            Contour.from_lines(lines=lines)

    def test_construct_from_lines_fails_last(self):
        lines = [[[0, 0], [2, 2]], [[2, 2], [5, 5]], [[5, 5], [0, 1]]]
        with pytest.raises(ValueError):
            Contour.from_lines(lines=lines)


class TestContourAddPoint:
    def test_add_point(self):
        cnt = Contour([[0, 0], [1, 1], [1, 0]])
        pt = [0, 1]
        assert np.array_equal(
            cnt.add_point(point=pt, index=1).asarray, [[0, 0], [0, 1], [1, 1], [1, 0]]
        )

    def test_add_point_last(self):
        cnt = Contour([[0, 0], [1, 1], [1, 0]])
        pt = [0, 1]
        assert np.array_equal(
            cnt.add_point(point=pt, index=-1).asarray, [[0, 0], [1, 1], [1, 0], [0, 1]]
        )

    def test_add_point_first_neg(self):
        cnt = Contour([[0, 0], [1, 1], [1, 0]])
        pt = [0, 1]
        assert np.array_equal(
            cnt.add_point(point=pt, index=-4).asarray, [[0, 1], [0, 0], [1, 1], [1, 0]]
        )

    def test_add_point_first(self):
        cnt = Contour([[0, 0], [1, 1], [1, 0]])
        pt = [0, 1]
        assert np.array_equal(
            cnt.add_point(point=pt, index=0).asarray, [[0, 1], [0, 0], [1, 1], [1, 0]]
        )

    def test_add_point_index_too_big(self):
        cnt = Contour([[0, 0], [1, 1], [1, 0]])
        pt = [0, 1]
        with pytest.raises(ValueError):
            cnt.add_point(point=pt, index=4)

    def test_add_point_index_too_small(self):
        cnt = Contour([[0, 0], [1, 1], [1, 0]])
        pt = [0, 1]
        with pytest.raises(ValueError):
            cnt.add_point(point=pt, index=-5)


class TestContourRearrange:
    def test_rearrange_first_point_at_index_pos(self):
        cnt = Contour([[0, 0], [1, 1], [1, 0]])
        assert np.array_equal(
            cnt.rearrange_first_point_at_index(index=1).asarray,
            [[1, 1], [1, 0], [0, 0]],
        )

    def test_rearrange_first_point_at_index_neg(self):
        cnt = Contour([[0, 0], [1, 1], [1, 0]])
        assert np.array_equal(
            cnt.rearrange_first_point_at_index(index=-2).asarray,
            [[1, 1], [1, 0], [0, 0]],
        )

    def test_rearrange_first_point_at_index_too_big(self):
        cnt = Contour([[0, 0], [1, 1], [1, 0]])
        with pytest.raises(ValueError):
            cnt.rearrange_first_point_at_index(index=3)

    def test_rearrange_first_point_at_index_too_small(self):
        cnt = Contour([[0, 0], [1, 1], [1, 0]])
        with pytest.raises(ValueError):
            cnt.rearrange_first_point_at_index(index=-4)


class TestContourFromUnorderedLinesApprox:
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
        cnt = Contour.from_unordered_lines_approx(input)
        assert np.all([o in output for o in cnt.lines.tolist()])

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
            Contour.from_unordered_lines_approx(input, max_dist_thresh=10)
