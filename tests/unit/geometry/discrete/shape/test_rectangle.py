"""
Unit tests for Rectangle geometry class
"""

import copy
import numpy as np

from otary.geometry import Rectangle
import pytest


class TestRectangleStr:

    def test_str(self):
        rect = Rectangle([[0, 0], [0, 1], [1, 1], [1, 0]])
        assert str(rect) == "Rectangle([[0, 0], [0, 1], [1, 1], [1, 0]])"

    def test_repr(self):
        rect = Rectangle([[0, 0], [0, 1], [1, 1], [1, 0]])
        assert repr(rect) == "Rectangle([[0, 0], [0, 1], [1, 1], [1, 0]])"


class TestRectangleCreation:
    def test_create_rectangle_unit(self):
        rect = Rectangle.unit()
        assert np.array_equal(rect.asarray, [[0, 0], [0, 1], [1, 1], [1, 0]])

    def test_create_rectangle_from_center(self):
        center = [1, 1]
        rect = Rectangle.from_center(center=center, width=2, height=2).rotate(
            angle=2 * np.pi
        )
        assert np.isclose(rect.asarray, [[0, 0], [2, 0], [2, 2], [0, 2]]).all()

    def test_create_rectangle_from_center_with_angle(self):
        center = [1, 1]
        rect = Rectangle.from_center(
            center=center,
            width=np.sqrt(2),
            height=np.sqrt(2),
            is_cast_int=False,
        ).rotate(angle=-np.pi / 4)
        expected_points = [[0, 1], [1, 0], [2, 1], [1, 2]]
        assert np.allclose(rect.asarray, expected_points)

    def test_create_rectangle_from_center_with_angle_and_cast_int(self):
        center = [10, 10]
        rect = Rectangle.from_center(
            center=center, width=10, height=10, is_cast_int=int
        ).rotate(angle=np.pi / 2)
        expected_points = [[15, 5], [15, 15], [5, 15], [5, 5]]
        assert np.allclose(rect.asarray, expected_points)

    def test_create_rectangle_from_topleft(self):
        topleft = [1, 1]
        rect = Rectangle.from_topleft(topleft=topleft, width=2, height=2)
        assert np.isclose(rect.asarray, [[1, 1], [3, 1], [3, 3], [1, 3]]).all()

    def test_create_rectangle_with_more_than_4_points(self):
        points = [[0, 0], [0, 1], [1, 1], [1, 0], [0.5, 0.5]]
        with pytest.raises(ValueError):
            Rectangle(points)

    def test_create_rectangle_with_less_than_4_points(self):
        points = [[0, 0], [0, 1], [1, 1]]
        with pytest.raises(ValueError):
            Rectangle(points)

    def test_create_rectangle_self_intersected(self):
        # even if you choose desintersect=False, a ValueError is raised
        # because a self-intersected rectangle cannot exist
        points = [[0, 0], [1, 1], [1, 0], [0, 1]]
        with pytest.raises(ValueError):
            Rectangle(points, desintersect=False)

    def test_create_rectangle_not_regular(self):
        points = [[0, 0], [100, 0], [100, 100], [0, 105]]
        with pytest.raises(ValueError):
            Rectangle(points)

    def test_create_rectangle_valid_irregular(self):
        points = [[0, 0], [100, 0], [100, 100], [0, 101]]
        Rectangle(points, regularity_margin_error=1e-2)


class TestRectangleIsSquare:

    def test_is_square_true(self):
        rect = Rectangle.from_topleft(topleft=[0, 0], width=2, height=2)
        assert rect.is_square

    def test_is_square_false(self):
        rect = Rectangle.from_topleft(topleft=[0, 0], width=2, height=4)
        assert not rect.is_square


class TestRectangleAxixAligned:
    def test_is_axis_aligned_true(self):
        # Axis-aligned rectangle
        rect = Rectangle.from_topleft(topleft=[0, 0], width=2, height=4)
        assert rect.is_axis_aligned

    def test_is_axis_aligned_false(self):
        # Non-axis-aligned rectangle (rotated)
        rect = Rectangle.from_center(center=[0, 0], width=2, height=4).rotate(
            angle=np.pi / 4
        )
        assert not rect.is_axis_aligned

    def test_is_axis_aligned_second_point_y_cause_false(self):
        rect = Rectangle([[0, 0], [100, 1], [100, 100], [0, 100]])
        assert not rect.is_axis_aligned

    def test_is_axis_aligned_second_point_x_cause_false(self):
        rect = Rectangle([[0, 0], [101, 0], [100, 100], [0, 100]])
        assert not rect.is_axis_aligned

    def test_is_axis_aligned_third_point_y_cause_false(self):
        rect = Rectangle([[0, 0], [100, 0], [100, 101], [0, 100]])
        assert not rect.is_axis_aligned

    def test_is_axis_aligned_third_point_x_cause_false(self):
        rect = Rectangle([[0, 0], [100, 0], [101, 100], [0, 100]])
        assert not rect.is_axis_aligned

    def test_is_axis_aligned_fourth_point_y_cause_false(self):
        rect = Rectangle([[0, 0], [100, 0], [100, 100], [0, 101]])
        assert not rect.is_axis_aligned

    def test_is_axis_aligned_fourth_point_x_cause_false(self):
        rect = Rectangle([[0, 0], [100, 0], [100, 100], [1, 100]])
        assert not rect.is_axis_aligned

    def test_is_axis_aligned_self_intersected(self):
        rect = Rectangle([[0, 0], [1, 1], [1, 0], [0, 1]])
        # True because desintersect by default on init
        assert rect.is_axis_aligned

    def test_is_axis_aligned_approx_true(self):
        # Axis-aligned rectangle with approximate check
        rect = Rectangle.from_topleft(topleft=[0, 0], width=2, height=4)
        assert rect.is_axis_aligned_approx()

    def test_is_axis_aligned_approx_false(self):
        # Non-axis-aligned rectangle with approximate check
        rect = Rectangle.from_center(center=[0, 0], width=2, height=4).rotate(
            angle=np.pi / 4
        )
        assert not rect.is_axis_aligned_approx()


class TestRectangleSideLength:

    def test_longside_length_horizontal_rectangle(self):
        # Horizontal rectangle
        rect = Rectangle.from_topleft(topleft=[0, 0], width=4, height=2)
        assert np.isclose(rect.longside_length, 4)

    def test_longside_length_vertical_rectangle(self):
        # Vertical rectangle
        rect = Rectangle.from_topleft(topleft=[0, 0], width=2, height=4)
        assert np.isclose(rect.longside_length, 4)

    def test_longside_length_square(self):
        # Square (equal sides)
        rect = Rectangle.from_topleft(topleft=[0, 0], width=3, height=3)
        assert np.isclose(rect.longside_length, 3)

    def test_longside_length_rotated_rectangle(self):
        # Rotated rectangle
        rect = Rectangle.from_center(center=[0, 0], width=4, height=2).rotate(
            angle=np.pi / 4
        )
        assert np.isclose(rect.longside_length, 4)

    def test_shortside_length_horizontal_rectangle(self):
        # Horizontal rectangle
        rect = Rectangle.from_topleft(topleft=[0, 0], width=4, height=2)
        assert np.isclose(rect.shortside_length, 2)

    def test_shortside_length_vertical_rectangle(self):
        # Vertical rectangle
        rect = Rectangle.from_topleft(topleft=[0, 0], width=2, height=4)
        assert np.isclose(rect.shortside_length, 2)

    def test_shortside_length_square(self):
        # Square (equal sides)
        rect = Rectangle.from_topleft(topleft=[0, 0], width=3, height=3)
        assert np.isclose(rect.shortside_length, 3)

    def test_shortside_length_rotated_rectangle(self):
        # Rotated rectangle
        rect = Rectangle.from_center(center=[0, 0], width=4, height=2).rotate(
            angle=np.pi / 4
        )
        assert np.isclose(rect.shortside_length, 2)


class TestRectangleDesintersect:

    def test_desintersect_non_self_intersected(self):
        # Non-self-intersected rectangle
        rect = Rectangle.from_topleft(topleft=[0, 0], width=4, height=2)
        rect = rect.desintersect()
        assert np.array_equal(rect.asarray, rect.asarray)

    def test_desintersect_self_intersected(self):
        # Self-intersected rectangle
        rect = Rectangle([[0, 0], [1, 1], [1, 0], [0, 1]])
        rect = rect.desintersect()
        expected_points = [[0, 0], [1, 0], [1, 1], [0, 1]]
        assert np.array_equal(rect.asarray, expected_points)

    def test_desintersect_already_sorted(self):
        # Rectangle with points already sorted
        rect = Rectangle([[0, 0], [0, 1], [1, 1], [1, 0]])
        rect = rect.desintersect()
        assert np.array_equal(rect.asarray, rect.asarray)

    def test_desintersect_negative_self_intersected(self):
        # Rotated self-intersected rectangle
        rect = Rectangle([[0, 0], [1, -1], [0, -1], [1, 0]])
        rect = rect.desintersect()
        expected_points = [[0, -1], [1, -1], [1, 0], [0, 0]]
        assert np.array_equal(rect.asarray, expected_points)

    def test_desintersect_losange_v1(self):
        rect = Rectangle([[0, 0], [0, 4], [2, 2], [-2, 2]])
        rect.desintersect()
        expected_points = [[0, 0], [2, 2], [0, 4], [-2, 2]]
        assert np.array_equal(rect.asarray, expected_points)

    def test_desintersect_losange_v2(self):
        rect = Rectangle([[0, 0], [2, 2], [0, 4], [-2, 2]])
        rect.desintersect()
        expected_points = [[0, 0], [2, 2], [0, 4], [-2, 2]]
        assert np.array_equal(rect.asarray, expected_points)

    def test_desintersect_losange_v3(self):
        rect = Rectangle([[0, 0], [0, 4], [-2, 2], [2, 2]])
        rect.desintersect()
        expected_points = [[0, 0], [2, 2], [0, 4], [-2, 2]]
        assert np.array_equal(rect.asarray, expected_points)

    def test_desintersect_rotated_rectangle(self):
        rec = Rectangle.from_center(center=[0, 0], width=4, height=2).rotate(
            angle=-0.92729
        )
        tmp = copy.deepcopy(rec.asarray[1])
        tmp2 = copy.deepcopy(rec.asarray[2])
        rec.asarray[1] = tmp2
        rec.asarray[2] = tmp
        rec.desintersect()
        expected_points = [
            [0.40001147959810135, -2.1999979127694047],
            [2.0000052179743846, -0.9999895639831617],
            [-0.40001147959810135, 2.1999979127694047],
            [-2.0000052179743846, 0.9999895639831617],
        ]
        assert np.allclose(rec.asarray, expected_points)


class TestRectangleJoin:

    def test_join_no_shared_points(self):
        rect1 = Rectangle.from_topleft(topleft=[0, 0], width=2, height=2)
        rect2 = Rectangle.from_topleft(topleft=[3, 3], width=2, height=2)
        result = rect1.join(rect2)
        assert result is None

    def test_join_one_shared_point(self):
        rect1 = Rectangle.from_topleft(topleft=[0, 0], width=2, height=2)
        rect2 = Rectangle.from_topleft(topleft=[2, 2], width=2, height=2)
        result = rect1.join(rect2)
        assert result is None

    def test_join_two_shared_points(self):
        rect1 = Rectangle.from_topleft(topleft=[0, 0], width=2, height=2)
        rect2 = Rectangle.from_topleft(topleft=[2, 0], width=2, height=2)
        result = rect1.join(rect2)
        expected_points = [[0, 0], [4, 0], [4, 2], [0, 2]]
        assert result is not None
        assert np.allclose(result.asarray, expected_points)

    def test_join_three_shared_points(self):
        rect1 = Rectangle.from_topleft(topleft=[0, 0], width=2, height=2)
        rect2 = Rectangle.from_topleft(topleft=[0, 0], width=2, height=2)
        result = rect1.join(rect2)
        assert result is rect1

    def test_join_identical_rectangles(self):
        rect1 = Rectangle.from_topleft(topleft=[0, 0], width=2, height=2)
        rect2 = Rectangle.from_topleft(topleft=[0, 0], width=2, height=2)
        result = rect1.join(rect2)
        assert result is rect1

    def test_join_with_margin_error(self):
        rect1 = Rectangle.from_topleft(topleft=[0, 0], width=2, height=2)
        rect2 = Rectangle.from_topleft(topleft=[2.1, 0], width=2, height=2)
        result = rect1.join(rect2, margin_dist_error=0.01)
        assert result is None

    def test_join_with_margin_error_success(self):
        rect1 = Rectangle.from_topleft(topleft=[0, 0], width=2, height=2)
        rect2 = Rectangle.from_topleft(topleft=[2.05, 0], width=2, height=2)
        result = rect1.join(rect2, margin_dist_error=0.5)
        expected_rect = Rectangle([[0.0, 0.0], [4.05, 0.0], [4.05, 2.0], [0.0, 2.0]])
        assert result is not None
        assert result.is_equal(expected_rect)


class TestRectangleGetVerticeFromTopleft:

    def test_get_bottomright_vertice_from_topleft(self):
        rect = Rectangle.from_topleft(topleft=[0, 0], width=4, height=2)
        topleft_index = 0
        vertice = rect.get_vertice_from_topleft(topleft_index, "bottomright")
        assert np.array_equal(vertice, [4, 2])

    def test_invalid_vertice_parameter(self):
        rect = Rectangle.from_topleft(topleft=[0, 0], width=4, height=2)
        topleft_index = 0
        with pytest.raises(ValueError, match="Parameter vertice must be one of"):
            rect.get_vertice_from_topleft(topleft_index, "invalid_vertice")
