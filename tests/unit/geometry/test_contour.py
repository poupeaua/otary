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
    def test_contour_is_regular(self):
        cnt1 = Contour([[0, 0], [1, 0], [1, 1], [0, 1]])
        assert cnt1.is_regular(margin_area_error_pct=0)

    def test_contour_is_regular_margin_error(self):
        cnt1 = Contour([[0, 0], [100, 0], [100, 101.2], [0, 100]])
        assert cnt1.is_regular(margin_area_error_pct=0.02)

    def test_contour_is_not_regular(self):
        cnt1 = Contour([[0, 0], [100, 0], [101, 100], [0, 100]])
        assert not cnt1.is_regular(margin_area_error_pct=0)

    def test_contour_is_not_regular_more_than_four_points(self):
        cnt1 = Contour([[0, 0], [1, 0], [1, 1], [0, 0.5], [0.5, 0.5]])
        assert not cnt1.is_regular(margin_area_error_pct=0)

    def test_contour_is_not_regular_less_than_four_points(self):
        cnt1 = Contour([[0, 0], [1, 0], [1.5, 1]])
        assert not cnt1.is_regular(margin_area_error_pct=5)


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
        points = Contour.reduce_collinear(points=input)
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
                [[0, 0], [100, 100], [500, 500], [0, 250], [0, 2], [0, 3], [1, 5]],
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
    def test_contour_reduce_by_distance_limit(self, input, output):
        points = Contour.reduce_by_distance_limit_n_successive_deletion(
            points=input, min_dist_threshold=5
        )
        assert np.array_equal(points, output)
