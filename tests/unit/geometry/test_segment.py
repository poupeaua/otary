"""
Test geometry file
"""

import numpy as np

from src.geometry import Segment


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
    def test_are_points_collinear_trivial(self):
        p1 = [0, 0]
        p2 = [0, 1]
        p3 = [0, 2]
        assert Segment.is_points_collinear(p1, p2, p3) is True

    def test_are_points_collinear_false(self):
        p1 = [0, 0]
        p2 = [7, 1]
        p3 = [0, 2]
        assert Segment.is_points_collinear(p1, p2, p3) is False

    def test_are_points_collinear_very_close(self):
        p1 = [-0.05, 0]
        p2 = [0, 1.01]
        p3 = [0.001, 2.02]
        assert Segment.is_points_collinear(p1, p2, p3) is True

    def test_are_points_collinear_spaced_points(self):
        p1 = [0, 0]
        p2 = [0, 1000]
        p3 = [0, 30000]
        assert Segment.is_points_collinear(p1, p2, p3) is True

    def test_are_points_collinear_spaced_points_false(self):
        p1 = [0, 15000]
        p2 = [50000, 1000]
        p3 = [-500, 30000]
        assert Segment.is_points_collinear(p1, p2, p3) is False

    def test_are_points_collinear_2_points_equal(self):
        p1 = [0, 0]
        p2 = [0, 0]
        p3 = [-500, 30000]
        assert Segment.is_points_collinear(p1, p2, p3) is True

    def test_are_points_collinear_2_close_points_false(self):
        p1 = [0, 1]
        p2 = [1, 2]
        p3 = [-500, 30000]
        assert Segment.is_points_collinear(p1, p2, p3) is False

    def test_are_points_collinear_2_very_close_points_true(self):
        p1 = [0, 1]
        p2 = [0.05, 0.99]
        p3 = [-500, 30000]
        assert Segment.is_points_collinear(p1, p2, p3) is False

    def test_are_points_collinear_3_points_equal(self):
        p1 = [0, 0]
        p2 = [0, 0]
        p3 = [0, 0]
        assert Segment.is_points_collinear(p1, p2, p3) is True


class TestAreLinesCollinear:
    def test_are_lines_collinear_equal(self):
        # case with two segments equal
        seg1 = Segment([[-1, 0], [-2, 0.01]])
        seg2 = Segment([[-1, 0], [-2, 0.01]])
        assert seg1.is_collinear(seg2) is True

    def test_are_lines_collinear_out(self):
        # out case or space-separated lines
        seg1 = Segment([[0, 0], [1, 1]])
        seg2 = Segment([[300, 300], [1000, 1000]])
        assert seg1.is_collinear(seg2) is True

    def test_are_lines_collinear_sup(self):
        # lines superposed case
        seg1 = Segment([[0, 0], [500, 500]])
        seg2 = Segment([[300, 300], [1000, 1000]])
        assert seg1.is_collinear(seg2) is True

    def test_are_lines_collinear_in(self):
        # a segment is bigger and envelop the smallest one
        seg1 = Segment([[0, 0], [500, 500]])
        seg2 = Segment([[300, 300], [400, 400]])
        assert seg1.is_collinear(seg2) is True

    def test_are_lines_collinear_only_parallel(self):
        # the two segments are just parallel but not collinear
        seg1 = Segment([[1, 0], [2, 1]])
        seg2 = Segment([[0, 1], [1, 2]])
        assert seg1.is_collinear(seg2) is False

    def test_are_lines_collinear_3_points_collinear(self):
        # three points are points collinear but the segments are not collinear
        seg1 = Segment([[0, 0], [2, 2]])
        seg2 = Segment([[1, 1], [-355, 56]])
        assert seg1.is_collinear(seg2) is False
