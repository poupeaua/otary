"""
Test Vector class util
"""

import numpy as np
import pytest

from otary.geometry import Vector


class TestVectorCardinal:
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

    def test_cardinal_direction_invalid_zero(self):
        with pytest.raises(ValueError):
            Vector([[0, 0], [0, -1]]).cardinal_direction(full=False, level=0)

    def test_cardinal_direction_invalid_four(self):
        with pytest.raises(ValueError):
            Vector([[0, 0], [0, -1]]).cardinal_direction(full=False, level=4)

    def test_cardinal_direction_N_lvl1(self):
        assert Vector([[0, 0], [0, -1]]).cardinal_direction(full=False, level=1) == "N"

    def test_cardinal_direction_E_lvl1(self):
        assert Vector([[0, 0], [1, 0]]).cardinal_direction(full=False, level=1) == "E"

    def test_cardinal_direction_S_lvl1(self):
        assert Vector([[0, 0], [0, 1]]).cardinal_direction(full=False, level=1) == "S"

    def test_cardinal_direction_W_lvl1(self):
        assert Vector([[0, 0], [-1, 0]]).cardinal_direction(full=False, level=1) == "W"

    def test_cardinal_direction_N(self):
        assert Vector([[0, 0], [0, -1]]).cardinal_direction(full=False, level=2) == "N"

    def test_cardinal_direction_NE(self):
        assert Vector([[0, 0], [1, -1]]).cardinal_direction(full=False, level=2) == "NE"

    def test_cardinal_direction_E(self):
        assert Vector([[0, 0], [1, 0]]).cardinal_direction(full=False, level=2) == "E"

    def test_cardinal_direction_SE(self):
        assert Vector([[0, 0], [1, 1]]).cardinal_direction(full=False, level=2) == "SE"

    def test_cardinal_direction_S(self):
        assert Vector([[0, 0], [0, 1]]).cardinal_direction(full=False, level=2) == "S"

    def test_cardinal_direction_SW(self):
        assert Vector([[0, 0], [-1, 1]]).cardinal_direction(full=False, level=2) == "SW"

    def test_cardinal_direction_W(self):
        assert Vector([[0, 0], [-1, 0]]).cardinal_direction(full=False, level=2) == "W"

    def test_cardinal_direction_NW(self):
        assert (
            Vector([[0, 0], [-1, -1]]).cardinal_direction(full=False, level=2) == "NW"
        )

    def test_cardinal_direction_full_N(self):
        assert (
            Vector([[0, 0], [0, -1]]).cardinal_direction(full=True, level=2) == "North"
        )

    def test_cardinal_direction_full_E(self):
        assert Vector([[0, 0], [1, 0]]).cardinal_direction(full=True, level=2) == "East"

    def test_cardinal_direction_full_S(self):
        assert (
            Vector([[0, 0], [0, 1]]).cardinal_direction(full=True, level=2) == "South"
        )

    def test_cardinal_direction_full_W(self):
        assert (
            Vector([[0, 0], [-1, 0]]).cardinal_direction(full=True, level=2) == "West"
        )
