"""
Unit tests for Rectangle geometry class
"""

import numpy as np

from src.geometry import Rectangle
import pytest
import pymupdf


class TestRectangleCreation:
    def test_create_rectangle_unit(self):
        rect = Rectangle.unit()
        assert np.array_equal(rect.asarray, [[0, 0], [0, 1], [1, 1], [1, 0]])

    def test_create_rectangle_from_center(self):
        center = [1, 1]
        rect = Rectangle.from_center(center=center, width=2, height=2, angle=2 * np.pi)
        assert np.isclose(rect.asarray, [[0, 2], [2, 2], [2, 0], [0, 0]]).all()

    def test_create_rectangle_from_topleft(self):
        topleft = [1, 1]
        rect = Rectangle.from_topleft(topleft=topleft, width=2, height=2)
        assert np.isclose(rect.asarray, [[1, 3], [3, 3], [3, 1], [1, 1]]).all()


class TestRectangleProperties:
    def test_is_axis_aligned_true(self):
        # Axis-aligned rectangle
        rect = Rectangle.from_topleft(topleft=[0, 0], width=2, height=4)
        assert rect.is_axis_aligned is True

    def test_is_axis_aligned_false(self):
        # Non-axis-aligned rectangle (rotated)
        rect = Rectangle.from_center(center=[0, 0], width=2, height=4, angle=np.pi / 4)
        assert rect.is_axis_aligned is False

    def test_is_axis_aligned_self_intersected(self):
        rect = Rectangle([[0, 0], [1, 1], [1, 0], [0, 1]])
        assert rect.is_axis_aligned is False

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

        # Assert that calling as_pymupdf_rect raises a ValueError
        with pytest.raises(RuntimeError):
            rect.as_pymupdf_rect

    def test_as_pymupdf_rect_self_intersected(self):
        # Create a self-intersected rectangle
        rect = Rectangle([[0, 0], [1, 1], [1, 0], [0, 1]])

        # Assert that calling as_pymupdf_rect raises a ValueError
        with pytest.raises(RuntimeError):
            rect.as_pymupdf_rect
