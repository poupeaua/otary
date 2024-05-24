"""
Test Vector class util
"""

import numpy as np

from src.geometry import Vector


class TestVector:
    def test_cardinal_degree_0(self):
        assert np.isclose(Vector([[0, 0], [0, -1]]).cardinal_degree, 0, atol=1e-7)

    def test_cardinal_degree_45(self):
        assert np.isclose(Vector([[0, 0], [1, -1]]).cardinal_degree, 45)

    def test_cardinal_degree_90(self):
        assert np.isclose(Vector([[0, 0], [1, 0]]).cardinal_degree, 90)

    def test_cardinal_degree_135(self):
        assert np.isclose(Vector([[0, 0], [1, 1]]).cardinal_degree, 135)

    def test_cardinal_degree_180(self):
        assert np.isclose(Vector([[0, 0], [0, 1]]).cardinal_degree, 180)

    def test_cardinal_degree_225(self):
        assert np.isclose(Vector([[0, 0], [-1, 1]]).cardinal_degree, 225)

    def test_cardinal_degree_270(self):
        assert np.isclose(Vector([[0, 0], [-1, 0]]).cardinal_degree, 270)

    def test_cardinal_degree_315(self):
        assert np.isclose(Vector([[0, 0], [-1, -1]]).cardinal_degree, 315)
