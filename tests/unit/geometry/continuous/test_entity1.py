"""
Tests for the ContinuousGeometryEntity class
"""

import numpy as np
from otary.geometry import Rectangle, Circle, Ellipse


class TestContinuousGeometryEntityBase:

    def test_n_points_polygonal_approx(self):
        entity = Circle(center=[0, 0], radius=1, n_points_polygonal_approx=1000)
        assert entity.n_points_polygonal_approx == 1000

    def test_setter_n_points_polygonal_approx_updates_value_and_polyapprox(
        self, mocker
    ):
        entity = Circle(center=[0, 0], radius=1, n_points_polygonal_approx=1000)
        mocker.spy(entity, "update_polyapprox")
        entity.n_points_polygonal_approx = 200
        assert entity.n_points_polygonal_approx == 200
        assert entity.update_polyapprox.call_count == 1


class TestContinuousGeometryEntityMaxMinCoordinates:
    def test_xmax_returns_maximum_x_of_polygonal_approx(self):
        entity = Ellipse(foci1=[4, 4], foci2=[4, 10], semi_major_axis=4)
        expected = entity.centroid[0] + entity.semi_minor_axis
        assert np.isclose(entity.xmax, expected)

    def test_xmin_returns_minimum_x_of_polygonal_approx(self):
        entity = Ellipse(foci1=[4, 4], foci2=[4, 10], semi_major_axis=4)
        expected = entity.centroid[0] - entity.semi_minor_axis
        assert np.isclose(entity.xmin, expected)

    def test_ymax_returns_maximum_y_of_polygonal_approx(self):
        entity = Ellipse(foci1=[4, 4], foci2=[4, 10], semi_major_axis=4)
        expected = entity.centroid[1] + entity.semi_major_axis
        assert np.isclose(entity.ymax, expected)

    def test_ymin_returns_minimum_y_of_polygonal_approx(self):
        entity = Ellipse(foci1=[4, 4], foci2=[4, 10], semi_major_axis=4)
        expected = entity.centroid[1] - entity.semi_major_axis
        assert np.isclose(entity.ymin, expected)


class TestContinuousGeometryEntityEnclosingAABB:

    def test_enclosing_axis_aligned_bbox_is_rectangle_instance(self):
        entity = Circle(center=[0, 0], radius=1, n_points_polygonal_approx=50)
        bbox = entity.enclosing_axis_aligned_bbox()
        assert isinstance(bbox, Rectangle)

    def test_enclosing_axis_aligned_bbox_for_centered_circle(self):
        # Circle at (0,0) with radius 2
        entity = Circle(center=[0, 0], radius=200, n_points_polygonal_approx=100)
        bbox = entity.enclosing_axis_aligned_bbox()
        expected_points = [[-200, -200], [200, -200], [200, 200], [-200, 200]]
        for point in expected_points:
            assert np.any(np.all(np.isclose(bbox.asarray, point, atol=1), axis=1))

    def test_enclosing_axis_aligned_bbox_for_offset_circle(self):
        # Circle at (5, 3) with radius 1.5
        entity = Circle(center=[5, 3], radius=1.5, n_points_polygonal_approx=100)
        bbox = entity.enclosing_axis_aligned_bbox()
        # Should be close to [3.5, 1.5], [6.5, 4.5]
        expected_points = [[3.5, 1.5], [6.5, 1.5], [6.5, 4.5], [3.5, 4.5]]
        for point in expected_points:
            assert np.any(np.all(np.isclose(bbox.asarray, point, atol=1), axis=1))
