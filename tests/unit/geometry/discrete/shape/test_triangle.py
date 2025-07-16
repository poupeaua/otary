"""
Unit tests for the Triangle class
"""

import pytest

from otary.geometry.discrete.shape.triangle import Triangle

class TestTriangleStr:

    def test_str(self):
        triangle = Triangle([[0, 0], [2, 0], [1, 2]])
        assert str(triangle) == "Triangle([[0, 0], [2, 0], [1, 2]])"

    def test_repr(self):
        triangle = Triangle([[0, 0], [2, 0], [1, 2]])
        assert repr(triangle) == "Triangle([[0, 0], [2, 0], [1, 2]])"


class TestInitTriangle:

    def test_init_more_than_three(self):
        with pytest.raises(ValueError):
            Triangle([[0, 0], [2, 0], [1, 2], [3, 4]])

    def test_init_less_than_three(self):
        with pytest.raises(ValueError):
            Triangle([[0, 0], [2, 0]])
        