"""
Test the Circle class.
"""

import numpy as np
from copy import copy
from otary.geometry import Circle, Polygon


class TestCircle:

    def setup_method(self):
        self.center = np.array([0, 0])
        self.radius = 1
        self.circle = Circle(center=self.center, radius=self.radius)

    def test_perimeter(self):
        assert self.circle.perimeter == 2 * np.pi * self.circle.radius

    def test_centroid(self):
        np.testing.assert_array_equal(self.circle.centroid, self.center)

    def test_shapely_surface(self):
        assert self.circle.shapely_surface is not None

    def test_shapely_edges(self):
        assert self.circle.shapely_edges is not None

    def test_polygonal_approx(self):
        n_points = 10
        polygon = self.circle.polygonal_approx(n_points=n_points)
        assert isinstance(polygon, Polygon)
        assert len(polygon.points) == n_points

    def test_curvature(self):
        assert self.circle.curvature() == 1 / self.radius

    def test_xmax(self):
        assert self.circle.xmax == self.center[0] + self.radius

    def test_xmin(self):
        assert self.circle.xmin == self.center[0] - self.radius

    def test_ymax(self):
        assert self.circle.ymax == self.center[1] + self.radius

    def test_ymin(self):
        assert self.circle.ymin == self.center[1] - self.radius

    def test_rotate_pivot(self):
        angle = np.pi / 2
        self.circle.rotate(angle=angle, pivot=[1, 1])
        np.testing.assert_array_almost_equal(self.circle.center, [2, 0])

    def test_rotate(self):
        angle = np.pi / 2
        copy_center = copy(self.circle.center)
        self.circle.rotate(angle=angle)
        np.testing.assert_array_almost_equal(self.circle.center, copy_center)

    def test_shift(self):
        vector = np.array([2, 3])
        pre_center = copy(self.circle.center)
        self.circle.shift(vector=vector)
        np.testing.assert_array_equal(self.circle.center, pre_center + vector)

    def test_normalize(self):
        x, y = 2, 3
        pre_center = copy(self.circle.center)
        self.circle.normalize(x=x, y=y)
        np.testing.assert_array_almost_equal(
            self.circle.center, pre_center / np.array([x, y])
        )

    def test_copy(self):
        copied_circle = self.circle.copy()
        assert copied_circle is not self.circle
        np.testing.assert_array_equal(copied_circle.center, self.circle.center)

    def test_str(self):
        assert str(self.circle) == f"Circle(center={self.center}, radius={self.radius})"

    def test_repr(self):
        assert (
            repr(self.circle) == f"Circle(center={self.center}, radius={self.radius})"
        )


class TestCircleIsCircle:

    def test_is_circle(self):
        assert Circle(center=[0, 0], radius=1).is_circle


class TestCircleEnclosingOBB:

    def test_enclosing_obb(self):
        circle = Circle(center=[0, 0], radius=100)
        bbox = circle.enclosing_oriented_bbox()
        expected_points = circle.enclosing_axis_aligned_bbox().asarray
        for point in expected_points:
            assert np.any(np.all(np.isclose(bbox.asarray, point, atol=1), axis=1))
