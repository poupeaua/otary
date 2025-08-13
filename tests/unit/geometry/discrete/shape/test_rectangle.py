"""
Unit tests for Rectangle geometry class
"""

import copy
import numpy as np

from otary.geometry import Rectangle
import pytest
import pymupdf


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
        rect = Rectangle.from_center(center=center, width=2, height=2, angle=2 * np.pi)
        assert np.isclose(rect.asarray, [[0, 0], [2, 0], [2, 2], [0, 2]]).all()

    def test_create_rectangle_from_center_with_angle(self):
        center = [1, 1]
        rect = Rectangle.from_center(
            center=center,
            width=np.sqrt(2),
            height=np.sqrt(2),
            angle=-np.pi / 4,
            is_cast_int=False,
        )
        expected_points = [[0, 1], [1, 0], [2, 1], [1, 2]]
        assert np.allclose(rect.asarray, expected_points)

    def test_create_rectangle_from_center_with_angle_and_cast_int(self):
        center = [10, 10]
        rect = Rectangle.from_center(
            center=center,
            width=10,
            height=10,
            angle=np.pi / 2,
            is_cast_int=int,
        )
        expected_points = [[15, 5], [15, 15], [5, 15], [5, 5]]
        assert np.allclose(rect.asarray, expected_points)

    def test_create_rectangle_from_topleft(self):
        topleft = [1, 1]
        rect = Rectangle.from_topleft(topleft=topleft, width=2, height=2)
        assert np.isclose(rect.asarray, [[1, 1], [3, 1], [3, 3], [1, 3]]).all()


class TestRectangleAxixAligned:
    def test_is_axis_aligned_true(self):
        # Axis-aligned rectangle
        rect = Rectangle.from_topleft(topleft=[0, 0], width=2, height=4)
        assert rect.is_axis_aligned == True

    def test_is_axis_aligned_false(self):
        # Non-axis-aligned rectangle (rotated)
        rect = Rectangle.from_center(center=[0, 0], width=2, height=4, angle=np.pi / 4)
        assert rect.is_axis_aligned == False

    def test_is_axis_aligned_self_intersected(self):
        rect = Rectangle([[0, 0], [1, 1], [1, 0], [0, 1]])
        assert rect.is_axis_aligned == False

    def test_is_axis_aligned_approx_true(self):
        # Axis-aligned rectangle with approximate check
        rect = Rectangle.from_topleft(topleft=[0, 0], width=2, height=4)
        assert rect.is_axis_aligned_approx() == True

    def test_is_axis_aligned_approx_false(self):
        # Non-axis-aligned rectangle with approximate check
        rect = Rectangle.from_center(center=[0, 0], width=2, height=4, angle=np.pi / 4)
        assert rect.is_axis_aligned_approx() == False


class TestRectanglePyMuRect:

    def test_as_pymupdf_rect_axis_aligned(self):
        # Create an axis-aligned rectangle
        rect = Rectangle.from_topleft(topleft=[0, 0], width=2, height=4)
        pymupdf_rect = rect.as_pymupdf_rect

        # Assert the pymupdf.Rect object has correct coordinates
        assert isinstance(pymupdf_rect, pymupdf.Rect)
        assert pymupdf_rect.x0 == 0
        assert pymupdf_rect.y0 == 0
        assert pymupdf_rect.x1 == 2
        assert pymupdf_rect.y1 == 4

    def test_as_pymupdf_rect_non_axis_aligned(self):
        # Create a non-axis-aligned rectangle
        rect = Rectangle.from_center(center=[0, 0], width=2, height=4, angle=np.pi / 4)

        # Assert that calling as_pymupdf_rect raises an error
        with pytest.raises(RuntimeError):
            rect.as_pymupdf_rect

    def test_as_pymupdf_rect_self_intersected(self):
        # Create a self-intersected rectangle
        rect = Rectangle([[0, 0], [1, 1], [1, 0], [0, 1]])

        # Assert that calling as_pymupdf_rect raises an error
        with pytest.raises(RuntimeError):
            rect.as_pymupdf_rect


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
        rect = Rectangle.from_center(center=[0, 0], width=4, height=2, angle=np.pi / 4)
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
        rect = Rectangle.from_center(center=[0, 0], width=4, height=2, angle=np.pi / 4)
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
        rec = Rectangle.from_center(center=[0, 0], width=4, height=2, angle=-0.92729)
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
        rect2 = Rectangle([[0, 0], [0, 2], [2, 2], [1, 1]])
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

    def test_invalid_vertice_parameter(self):
        rect = Rectangle.from_topleft(topleft=[0, 0], width=4, height=2)
        topleft_index = 0
        with pytest.raises(ValueError, match="Parameter vertice must be one of"):
            rect.get_vertice_from_topleft(topleft_index, "invalid_vertice")
