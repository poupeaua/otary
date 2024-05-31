"""
Unit tests for Rectangle geometry class
"""

import numpy as np

from src.geometry import Rectangle


class TestContourIsEqual:
    def test_create_rectangle_from_center(self):
        center = [1, 1]
        rect = Rectangle.from_center(center=center, dim=(2, 2), angle=2 * np.pi)
        assert np.isclose(rect.asarray, [[0, 2], [2, 2], [2, 0], [0, 0]]).all()
