"""
Unit tests for LinearSpline geometry class
"""

import numpy as np
import pytest

from otary.geometry.discrete.linear.linear_spline import LinearSpline


class TestLinearSplineInit:

    def test_init(self):
        points = np.array([[0, 0], [1, 0], [2, 0]])
        ls = LinearSpline(points=points)
        assert np.array_equal(ls.asarray, points)

    def test_single_point_error(self):
        # Degenerate case: only one point
        points = np.array([[3, 4]])
        with pytest.raises(ValueError):
            LinearSpline(points=points)


class TestLinearSplineCentroid:

    def test_centroid_straight_line(self):
        # For a straight line from (0,0) to (2,0), centroid should be at (1,0)
        points = np.array([[0, 0], [2, 0]])
        ls = LinearSpline(points=points)
        centroid = ls.centroid
        np.testing.assert_allclose(centroid, [1, 0])

    def test_centroid_polyline(self):
        # For a polyline: (0,0)-(2,0)-(2,2)
        points = np.array([[0, 0], [2, 0], [2, 2]])
        ls = LinearSpline(points=points)
        # Manually compute expected centroid
        # Segment 1: (0,0)-(2,0): mid=(1,0), length=2
        # Segment 2: (2,0)-(2,2): mid=(2,1), length=2
        # Weighted centroid: ((1*2)+(2*2), (0*2)+(1*2)) / (2+2) = (2+4, 0+2)/4 = (6/4, 2/4) = (1.5, 0.5)
        expected = [1.5, 0.5]
        np.testing.assert_allclose(ls.centroid, expected)

    def test_centroid_two_points_diagonal(self):
        # Diagonal line from (0,0) to (1,1), centroid should be at (0.5, 0.5)
        points = np.array([[0, 0], [1, 1]])
        ls = LinearSpline(points=points)
        np.testing.assert_allclose(ls.centroid, [0.5, 0.5])

    def test_centroid_zero_length_segments(self):
        # All points are the same, centroid should be that point
        points = np.array([[1, 2], [1, 2], [1, 2]])
        ls = LinearSpline(points=points)
        np.testing.assert_allclose(ls.centroid, [1, 2])


class TestLinearSplineMidpoint:

    def test_midpoint_straight_line(self):
        # For a straight line from (0,0) to (2,0), midpoint should be at (1,0)
        points = np.array([[0, 0], [2, 0]])
        ls = LinearSpline(points=points)
        np.testing.assert_allclose(ls.midpoint, [1, 0])

    def test_midpoint_polyline(self):
        # For a polyline: (0,0)-(2,0)-(2,2)
        points = np.array([[0, 0], [2, 0], [2, 2]])
        ls = LinearSpline(points=points)
        # Total length = 2 + 2 = 4, midpoint at length 2
        # First segment: (0,0)-(2,0), length=2
        # So midpoint is at the end of first segment: (2,0)
        np.testing.assert_allclose(ls.midpoint, [2, 0])

    def test_midpoint_two_points_diagonal(self):
        # Diagonal line from (0,0) to (1,1), midpoint should be at (0.5, 0.5)
        points = np.array([[0, 0], [1, 1]])
        ls = LinearSpline(points=points)
        np.testing.assert_allclose(ls.midpoint, [0.5, 0.5])

    def test_midpoint_zero_length_segments(self):
        # All points are the same, midpoint should be that point
        points = np.array([[1, 2], [1, 2], [1, 2]])
        ls = LinearSpline(points=points)
        np.testing.assert_allclose(ls.midpoint, [1, 2])
