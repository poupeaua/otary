"""
Test the Ellipse class.
"""

import pytest
import math
import numpy as np
from otary.geometry.continuous.shape.ellipse import Ellipse
from shapely.geometry import Polygon as SPolygon, LineString


class TestEllipse:

    def test_ellipse_initialization(self):
        foci1 = np.array([0, 0])
        foci2 = np.array([4, 0])
        semi_major_axis = 5
        ellipse = Ellipse(foci1, foci2, semi_major_axis)
        assert np.array_equal(ellipse.foci1, foci1)
        assert np.array_equal(ellipse.foci2, foci2)
        assert ellipse.semi_major_axis == semi_major_axis

    def test_ellipse_invalid_initialization(self):
        foci1 = np.array([0, 0])
        foci2 = np.array([10, 0])
        semi_major_axis = 4
        with pytest.raises(ValueError):
            Ellipse(foci1, foci2, semi_major_axis)

    def test_ellipse_properties(self):
        foci1 = np.array([0, 0])
        foci2 = np.array([4, 0])
        semi_major_axis = 5
        ellipse = Ellipse(foci1, foci2, semi_major_axis)
        expected_semi_minor_axis = math.sqrt(21)
        assert np.array_equal(ellipse.centroid, np.array([2, 0]))
        assert ellipse.semi_minor_axis == expected_semi_minor_axis
        assert ellipse.linear_eccentricity == 2
        assert ellipse.eccentricity == 2 / 5
        assert ellipse.area == pytest.approx(
            math.pi * semi_major_axis * expected_semi_minor_axis, rel=1e-2
        )

    def test_ellipse_rotation(self):
        foci1 = np.array([0, 2])
        foci2 = np.array([4, 2])
        semi_major_axis = 5
        ellipse = Ellipse(foci1, foci2, semi_major_axis)
        rotated_ellipse = ellipse.rotate(angle=90, is_degree=True)
        assert rotated_ellipse.foci1[0] == pytest.approx(2, rel=1e-2)
        assert rotated_ellipse.foci1[1] == pytest.approx(4, rel=1e-2)
        assert rotated_ellipse.foci2[0] == pytest.approx(2, rel=1e-2)
        assert rotated_ellipse.foci2[1] == pytest.approx(0, rel=1e-2)

    def test_ellipse_shift(self):
        foci1 = np.array([0, 0])
        foci2 = np.array([4, 0])
        semi_major_axis = 5
        ellipse = Ellipse(foci1, foci2, semi_major_axis)
        shifted_ellipse = ellipse.shift(vector=np.array([1, 1]))
        assert np.array_equal(shifted_ellipse.foci1, np.array([1, 1]))
        assert np.array_equal(shifted_ellipse.foci2, np.array([5, 1]))

    def test_ellipse_normalize(self):
        foci1 = np.array([0, 2])
        foci2 = np.array([4, 0])
        semi_major_axis = 5
        ellipse = Ellipse(foci1, foci2, semi_major_axis)
        normalized_ellipse = ellipse.normalize(x=4, y=4)
        assert np.array_equal(normalized_ellipse.foci1, np.array([0, 1 / 2]))
        assert np.array_equal(normalized_ellipse.foci2, np.array([1, 0]))

    def test_ellipse_perimeter_approx(self):
        foci1 = np.array([0, 0])
        foci2 = np.array([4, 0])
        semi_major_axis = 5
        ellipse = Ellipse(foci1, foci2, semi_major_axis)
        perimeter = ellipse.perimeter_approx()
        assert perimeter == pytest.approx(
            ellipse.perimeter_approx(is_ramanujan=True), rel=1e-2
        )

    def test_ellipse_polygonal_approx(self):
        foci1 = np.array([0, 0])
        foci2 = np.array([4, 0])
        semi_major_axis = 5
        ellipse = Ellipse(foci1, foci2, semi_major_axis)
        n_points = 10
        polygon = ellipse.polygonal_approx(n_points=n_points)
        assert len(polygon.points) == n_points

    def test_ellipse_copy(self):
        foci1 = np.array([0, 0])
        foci2 = np.array([4, 0])
        semi_major_axis = 5
        ellipse = Ellipse(foci1, foci2, semi_major_axis)
        ellipse_copy = ellipse.copy()
        assert np.array_equal(ellipse_copy.foci1, ellipse.foci1)
        assert np.array_equal(ellipse_copy.foci2, ellipse.foci2)
        assert ellipse_copy.semi_major_axis == ellipse.semi_major_axis

    def test_ellipse_focal_distance(self):
        foci1 = np.array([0, 0])
        foci2 = np.array([4, 0])
        semi_major_axis = 5
        ellipse = Ellipse(foci1, foci2, semi_major_axis)
        assert ellipse.focal_distance == 2

    def test_ellipse_shapely_surface(self):
        foci1 = np.array([0, 0])
        foci2 = np.array([4, 0])
        semi_major_axis = 5
        ellipse = Ellipse(foci1, foci2, semi_major_axis)
        shapely_surface = ellipse.shapely_surface
        assert isinstance(shapely_surface, SPolygon)
        assert shapely_surface.area == pytest.approx(ellipse.area, rel=1e-2)

    def test_ellipse_shapely_curve(self):
        foci1 = np.array([0, 0])
        foci2 = np.array([4, 0])
        semi_major_axis = 5
        ellipse = Ellipse(foci1, foci2, semi_major_axis)
        assert isinstance(ellipse.shapely_edges, LineString)
        assert ellipse.shapely_edges.length == pytest.approx(
            ellipse.perimeter_approx(), rel=1e-2
        )

    def test_ellipse_curvature(self):
        foci1 = np.array([0, 0])
        foci2 = np.array([0, 0])
        semi_major_axis = 5
        ellipse = Ellipse(foci1, foci2, semi_major_axis)
        curvature = ellipse.curvature(point=np.array([5, 0]))
        expected_curvature = 1 / semi_major_axis
        assert curvature == pytest.approx(expected_curvature, rel=1e-2)

    def test_ellipse_str(self):
        foci1 = np.array([0, 0])
        foci2 = np.array([4, 0])
        semi_major_axis = 5
        ellipse = Ellipse(foci1, foci2, semi_major_axis)
        assert str(ellipse) == "Ellipse(foci1=[0 0], foci2=[4 0], a=5)"

    def test_ellipse_repr(self):
        foci1 = np.array([0, 0])
        foci2 = np.array([4, 0])
        semi_major_axis = 5
        ellipse = Ellipse(foci1, foci2, semi_major_axis)
        assert repr(ellipse) == "Ellipse(foci1=[0 0], foci2=[4 0], a=5)"


class TestEllipseIsCircle:
    def test_is_circle_true(self):
        # Circle: foci coincide, a == b
        foci1 = np.array([0, 0])
        foci2 = np.array([0, 0])
        semi_major_axis = 5
        ellipse = Ellipse(foci1, foci2, semi_major_axis)
        assert ellipse.is_circle is True

    def test_is_circle_false(self):
        # Ellipse: foci do not coincide, a > b
        foci1 = np.array([0, 0])
        foci2 = np.array([4, 0])
        semi_major_axis = 5
        ellipse = Ellipse(foci1, foci2, semi_major_axis)
        assert ellipse.is_circle is False

    def test_is_circle_almost_circle(self):
        # Ellipse: foci very close, a ≈ b, but not exactly
        foci1 = np.array([0, 0])
        foci2 = np.array([1e-5, 0])
        semi_major_axis = 5
        ellipse = Ellipse(foci1, foci2, semi_major_axis)
        # Should be False because equality is strict
        assert ellipse.is_circle is False
