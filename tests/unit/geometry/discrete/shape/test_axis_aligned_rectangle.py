"""
Tests for the AxisAlignedRectangle class
"""

import pymupdf
import pytest

from otary.geometry import AxisAlignedRectangle, Rectangle


class TestAxisAlignedRectangleCreation:

    def test_create_axis_aligned_rectangle_base(self):
        # Create an axis-aligned rectangle
        rect = AxisAlignedRectangle.from_topleft(topleft=[0, 0], width=2, height=4)

        # Assert the rectangle has correct coordinates
        assert rect.xmin == 0
        assert rect.ymin == 0
        assert rect.xmax == 2
        assert rect.ymax == 4

        # assert first point is top-left and points are ordered clockwise
        assert (rect.points[0] == [0, 0]).all()
        assert (rect.points[1] == [2, 0]).all()
        assert (rect.points[2] == [2, 4]).all()
        assert (rect.points[3] == [0, 4]).all()

    def test_create_axis_aligned_rectangle_from_center(self):
        # Create an axis-aligned rectangle from center
        rect = AxisAlignedRectangle.from_center(center=[5, 5], width=4, height=2)

        # Assert the rectangle has correct coordinates
        assert rect.xmin == 3
        assert rect.ymin == 4
        assert rect.xmax == 7
        assert rect.ymax == 6

        # assert first point is top-left and points are ordered clockwise
        assert (rect.points[0] == [3, 4]).all()
        assert (rect.points[1] == [7, 4]).all()
        assert (rect.points[2] == [7, 6]).all()
        assert (rect.points[3] == [3, 6]).all()

    def test_create_non_axis_aligned_rectangle_raises(self):
        # Attempt to create a non-axis-aligned rectangle
        with pytest.raises(ValueError):
            rect = Rectangle.from_center(center=[2, 5], width=6, height=10).rotate(
                angle=30
            )
            AxisAlignedRectangle.from_rectangle(rectangle=rect)


class TestAxisAlignedRectangleProperties:

    def test_height_and_width_properties(self):
        # Create an axis-aligned rectangle
        rect = AxisAlignedRectangle.from_topleft(topleft=[0, 0], width=3, height=5)

        # Assert the height and width properties are correct
        assert rect.height == 5
        assert rect.width == 3

    def test_area_property(self):
        # Create an axis-aligned rectangle
        width = 3
        height = 5
        rect = AxisAlignedRectangle.from_topleft(topleft=[0, 0], width=3, height=5)

        # Assert the area property is correct
        assert rect.area == width * height

    def test_perimeter_property(self):
        # Create an axis-aligned rectangle
        width = 3
        height = 5
        rect = AxisAlignedRectangle.from_topleft(
            topleft=[0, 0], width=width, height=height
        )

        # Assert the perimeter property is correct
        assert rect.perimeter == 2 * (width + height)


class TestAxisAlignedRectanglePyMuRect:

    def test_as_pymupdf_rect_base(self):
        # Create an axis-aligned rectangle
        rect = AxisAlignedRectangle.from_topleft(topleft=[0, 0], width=2, height=4)
        pymupdf_rect = rect.as_pymupdf_rect

        # Assert the pymupdf.Rect object has correct coordinates
        assert isinstance(pymupdf_rect, pymupdf.Rect)
        assert pymupdf_rect.x0 == 0
        assert pymupdf_rect.y0 == 0
        assert pymupdf_rect.x1 == 2
        assert pymupdf_rect.y1 == 4


class TestAxisAlignedRectangleShift:

    def test_shift_base(self):
        # Create an axis-aligned rectangle
        rect = AxisAlignedRectangle.from_topleft(topleft=[1, 1], width=3, height=2)

        # Shift the rectangle by (2, 3)
        rect.shift(vector=[2, 3])

        # Assert the new coordinates are correct
        assert rect.xmin == 3
        assert rect.ymin == 4
        assert rect.xmax == 6
        assert rect.ymax == 6


class TestAxisAlignedRectangleRotated90:

    def test_rotated90_assert_axis_aligned(self):
        # Create an axis-aligned rectangle
        rect = AxisAlignedRectangle.from_topleft(topleft=[1, 1], width=3, height=2)

        # Rotate the rectangle by 90 degrees
        rect_rot = rect.rotated90
        assert rect_rot.is_axis_aligned

    def test_rotated90_assert_new_coords(self):
        # Create an axis-aligned rectangle
        rect = AxisAlignedRectangle.from_topleft(topleft=[1, 1], width=3, height=2)
        rect_rot = rect.rotated90

        # Assert the new coordinates are correct
        assert rect_rot.width == rect.height
        assert rect_rot.height == rect.width
        assert rect_rot.centroid[0] == rect.centroid[0]
        assert rect_rot.centroid[1] == rect.centroid[1]

    def test_rotated90_assert_topleft_and_clockwise(self):
        # Create an axis-aligned rectangle
        rect = AxisAlignedRectangle.from_topleft(topleft=[1, 1], width=3, height=2)
        rect_rot = rect.rotated90

        # assert first point is top-left and points are ordered clockwise
        assert (rect_rot.asarray[0] == [1.5, 0.5]).all()
        assert rect_rot.is_clockwise(is_y_axis_down=True)


class TestAxisAlignedRectangleRotateTransform:

    def test_rotate_transform_base(self):
        # Create an axis-aligned rectangle
        rect = AxisAlignedRectangle.from_topleft(topleft=[1, 1], width=3, height=2)

        # Rotate the rectangle by 45 degrees
        rect_rot = rect.rotate_transform(angle=45)

        assert not rect_rot.is_axis_aligned
        assert not isinstance(rect_rot, AxisAlignedRectangle)


class TestAxisAlignedRectangleRotate:

    def test_rotate_raises(self):
        # Create an axis-aligned rectangle
        rect = AxisAlignedRectangle.from_topleft(topleft=[1, 1], width=3, height=2)

        # Attempt to rotate the rectangle by 45 degrees
        with pytest.raises(TypeError):
            rect.rotate(angle=45)
