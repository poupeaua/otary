"""
Test geometry file
"""

import pytest
import numpy as np

from otary.geometry import Segment


class TestSegmentProperties:
    def test_perimeter(self):
        seg = Segment([[0, 0], [1, 0]])
        assert seg.perimeter == 1

    def test_length(self):
        seg = Segment([[0, 0], [1, 0]])
        assert seg.perimeter == 1

    def test_centroid(self):
        seg = Segment([[0, 0], [1, 0]])
        assert np.equal(seg.centroid, np.array([0.5, 0])).all()

    def test_slope_0(self):
        seg = Segment([[0, 0], [1, 0]])
        assert np.isclose(seg.slope, 0)

    def test_slope_1(self):
        seg = Segment([[0, 0], [1, 1]])
        assert np.isclose(seg.slope, 1)

    @pytest.mark.parametrize(
        "slope", [-1e10, -10, -5, -3, -1, -1 / 2, 0, 1 / 2, 1, 3, 5, 10, 1e10]
    )
    def test_slope_parametrized_slope(self, slope):
        seg = Segment([[0, 0], [1, slope]])
        assert np.isclose(seg.slope, slope)

    def test_slope_cv2(self):
        seg = Segment([[0, 0], [1, 1]])
        assert np.isclose(seg.slope_cv2, -1)


class TestSegmentSlopeCalculation:
    def test_compute_slope_angle_pos_xunit(self):
        seg = np.array([[0, 0], [1, 0]])
        assert Segment(seg).slope_angle() == 0

    def test_compute_slope_angle_neg_xunit(self):
        seg = np.array([[0, 0], [-1, 0]])
        assert Segment(seg).slope_angle() == 0

    def test_compute_slope_angle_pos_yunit(self):
        seg = np.array([[0, 0], [0, 1]])
        assert np.isclose(Segment(seg).slope_angle(), np.pi / 2)

    def test_compute_slope_angle_neg_yunit(self):
        seg = np.array([[0, 0], [0, -1]])
        assert np.isclose(Segment(seg).slope_angle(), -np.pi / 2)

    def test_compute_slope_angle_posx_pi_over_4(self):
        seg = np.array([[0, 0], [1, 1]])
        assert np.isclose(Segment(seg).slope_angle(), np.pi / 4)

    def test_compute_slope_angle_posx_neg_pi_over_4(self):
        seg = np.array([[0, 0], [1, -1]])
        assert np.isclose(Segment(seg).slope_angle(), -np.pi / 4)

    def test_compute_slope_angle_negx_pi_over_4(self):
        seg = np.array([[0, 0], [-1, 1]])
        assert np.isclose(Segment(seg).slope_angle(), -np.pi / 4)

    def test_compute_slope_angle_negx_neg_pi_over_4(self):
        seg = np.array([[0, 0], [-1, -1]])
        assert np.isclose(Segment(seg).slope_angle(), np.pi / 4)


class TestAreParallel:
    def test_are_parallel_perpendicular(self):
        seg1 = Segment([[0, 0], [0, 1]])
        seg2 = Segment([[0, 0], [1, 0]])
        assert seg1.is_parallel(seg2) is False

    def test_are_parallel_exactly_parallel(self):
        seg1 = Segment([[-1, 0], [-2, 0]])
        seg2 = Segment([[1, 0], [2, 0]])
        assert seg1.is_parallel(seg2) is True

    def test_are_parallel_very_close_to_parallel_inward(self):
        seg1 = Segment([[-1, 0], [-2, 0.01]])
        seg2 = Segment([[1, 0], [2, 0.01]])
        assert seg1.is_parallel(seg2) is True

    def test_are_parallel_very_close_to_parallel_outward(self):
        seg1 = Segment([[-1, 0], [-2, 0.01]])
        seg2 = Segment([[1, 0], [2, -0.01]])
        assert seg1.is_parallel(seg2) is True

    def test_are_parallel_spacely_separated(self):
        seg1 = Segment([[-1, 0], [-2, 0.01]])
        seg2 = Segment([[1, 0], [2, -0.01]])
        assert seg1.is_parallel(seg2) is True

    def test_are_parallel_same_line(self):
        seg1 = Segment([[3, 10], [5, 12]])
        seg2 = Segment([[-3, -10], [-5, -12]])
        assert seg1.is_parallel(seg2) is True


class TestArePointsCollinear:
    def test_is_points_collinear_trivial(self):
        p1 = [0, 0]
        p2 = [0, 1]
        p3 = [0, 2]
        assert Segment.is_points_collinear(p1, p2, p3) is True

    def test_is_points_collinear_false(self):
        p1 = [0, 0]
        p2 = [7, 1]
        p3 = [0, 2]
        assert Segment.is_points_collinear(p1, p2, p3) is False

    def test_is_points_collinear_very_close(self):
        p1 = [-0.05, 0]
        p2 = [0, 1.01]
        p3 = [0.001, 2.02]
        assert Segment.is_points_collinear(p1, p2, p3) is True

    def test_is_points_collinear_spaced_points(self):
        p1 = [0, 0]
        p2 = [0, 1000]
        p3 = [0, 30000]
        assert Segment.is_points_collinear(p1, p2, p3) is True

    def test_is_points_collinear_spaced_points_false(self):
        p1 = [0, 15000]
        p2 = [50000, 1000]
        p3 = [-500, 30000]
        assert Segment.is_points_collinear(p1, p2, p3) is False

    def test_is_points_collinear_2_points_equal(self):
        p1 = [0, 0]
        p2 = [0, 0]
        p3 = [-500, 30000]
        assert Segment.is_points_collinear(p1, p2, p3) is True

    def test_is_points_collinear_2_close_points_false(self):
        p1 = [0, 1]
        p2 = [1, 2]
        p3 = [-500, 30000]
        assert Segment.is_points_collinear(p1, p2, p3) is False

    def test_is_points_collinear_2_very_close_points_true(self):
        p1 = [0, 1]
        p2 = [0.05, 0.99]
        p3 = [-500, 30000]
        assert Segment.is_points_collinear(p1, p2, p3) is False

    def test_is_points_collinear_3_points_equal(self):
        p1 = [0, 0]
        p2 = [0, 0]
        p3 = [0, 0]
        assert Segment.is_points_collinear(p1, p2, p3) is True

    def test_is_point_collinear(self):
        p1 = [0, 0]
        p2 = [0, 1000]
        p3 = [0, 30000]
        seg = Segment(points=[p1, p2])
        assert seg.is_point_collinear(p3) is True


class TestAreLinesCollinear:
    def test_is_lines_collinear_equal(self):
        # case with two segments equal
        seg1 = Segment([[-1, 0], [-2, 0.01]])
        seg2 = Segment([[-1, 0], [-2, 0.01]])
        assert seg1.is_collinear(seg2) is True

    def test_is_lines_collinear_out(self):
        # out case or space-separated lines
        seg1 = Segment([[0, 0], [1, 1]])
        seg2 = Segment([[300, 300], [1000, 1000]])
        assert seg1.is_collinear(seg2) is True

    def test_is_lines_collinear_sup(self):
        # lines superposed case
        seg1 = Segment([[0, 0], [500, 500]])
        seg2 = Segment([[300, 300], [1000, 1000]])
        assert seg1.is_collinear(seg2) is True

    def test_is_lines_collinear_in(self):
        # a segment is bigger and envelop the smallest one
        seg1 = Segment([[0, 0], [500, 500]])
        seg2 = Segment([[300, 300], [400, 400]])
        assert seg1.is_collinear(seg2) is True

    def test_is_lines_collinear_only_parallel(self):
        # the two segments are just parallel but not collinear
        seg1 = Segment([[1, 0], [2, 1]])
        seg2 = Segment([[0, 1], [1, 2]])
        assert seg1.is_collinear(seg2) is False

    def test_is_lines_collinear_3_points_collinear(self):
        # three points are points collinear but the segments are not collinear
        seg1 = Segment([[0, 0], [2, 2]])
        seg2 = Segment([[1, 1], [-355, 56]])
        assert seg1.is_collinear(seg2) is False


class TestSegmentNormal:
    def test_normal_is_orthogonal(self):
        """Check that dot product between the normal and original vector is 0"""
        seg = Segment([[0, 0], [1, 0]])
        normal = seg.normal()
        # The direction vector of seg is [1, 0], normal should be [0, 1] or [0, -1]
        direction = seg.points[1] - seg.points[0]
        normal_direction = normal.points[1] - normal.points[0]
        dot_product = np.dot(direction, normal_direction)
        assert np.isclose(dot_product, 0)

    def test_normal_length_equals_original(self):
        seg = Segment([[0, 0], [3, 4]])
        normal = seg.normal()
        length = np.linalg.norm(seg.points[1] - seg.points[0])
        normal_length = np.linalg.norm(normal.points[1] - normal.points[0])
        assert np.isclose(length, normal_length)

    def test_normal_centroid_equals_original(self):
        seg = Segment([[2, 2], [4, 6]])
        normal = seg.normal()
        assert np.allclose(seg.centroid, normal.centroid)

    def test_normal_of_vertical_segment(self):
        seg = Segment([[0, 0], [0, 2]])
        normal = seg.normal()
        # The normal should be horizontal
        direction = normal.points[1] - normal.points[0]
        assert np.isclose(direction[1], 0)

    def test_normal_of_horizontal_segment(self):
        seg = Segment([[1, 5], [4, 5]])
        normal = seg.normal()
        # The normal should be vertical
        direction = normal.points[1] - normal.points[0]
        assert np.isclose(direction[0], 0)

    def test_normal_of_normal_is_equivalent(self):
        """Test that the normal of the normal segment is equivalent to the original segment (up to direction)."""
        seg = Segment([[1, 2], [4, 6]])
        normal = seg.normal()
        normal_of_normal = normal.normal()
        # The normal of the normal should be the original segment, possibly reversed
        # Compare sorted points to ignore direction
        orig_points_sorted = np.sort(seg.points, axis=0)
        normal2_points_sorted = np.sort(normal_of_normal.points, axis=0)
        assert np.allclose(orig_points_sorted, normal2_points_sorted)
