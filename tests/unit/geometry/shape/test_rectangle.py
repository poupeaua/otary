"""
Unit tests for Rectangle geometry class
"""

import numpy as np

from src.geometry import Rectangle


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
