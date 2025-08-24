"""
File containing tests for the VectorizedLinearSpline class.
"""

import numpy as np
import pytest

from otary.geometry.discrete.linear.directed.vectorized_linear_spline import VectorizedLinearSpline


class TestVectorizedLinearSplineInit:

    def test_init(self):
        points = np.array([[0, 0], [1, 0], [2, 0]])
        vls = VectorizedLinearSpline(points=points)
        assert np.array_equal(vls.asarray, points)

    def test_cardinal_degree_single_point_error(self):
        points = np.array([[0, 0]])
        with pytest.raises(ValueError):
            VectorizedLinearSpline(points=points)

class TestVectorizedLinearSplineIsSimpleVector:

    def test_is_simple_vector_false(self):
        points = np.array([[0, 0], [1, 0], [2, 0]])
        vls = VectorizedLinearSpline(points=points)
        assert not vls.is_simple_vector

    def test_is_simple_vector_true(self):
        points = np.array([[0, 0], [1, 0]])
        vls = VectorizedLinearSpline(points=points)
        assert vls.is_simple_vector

class TestVectorizedLinearSplineCardinalDegree:

    def test_cardinal_degree_base(self):
        points = np.array([[0, 0], [1, 0], [2, 0]])
        vls = VectorizedLinearSpline(points=points)
        assert vls.cardinal_degree == 90

    def test_cardinal_degree_base_other(self):
        points = np.array([[0, 0], [5, 5], [2, 0]])
        vls = VectorizedLinearSpline(points=points)
        assert vls.cardinal_degree == 90

    def test_cardinal_degree_two_points(self):
        points = np.array([[0, 0], [100, 100]])
        vls = VectorizedLinearSpline(points=points)
        assert round(vls.cardinal_degree + 90*10, 3) % 45 == 0
