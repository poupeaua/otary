"""
Test tools.py
"""

import numpy as np
import pytest

from otary.geometry.utils.tools import assert_list_of_lines
from otary.utils.tools import assert_transform_shift_vector
from otary.utils.tools import assert_transform_shift_vector


class TestAssertListOfLines:

    def test_assert_list_of_lines_valid(self):
        # Valid input: list of two lines, each line is a 2x2 array
        lines = np.asarray([[[0, 0], [1, 1]], [[2, 2], [3, 3]]])
        assert_list_of_lines(lines)  # Should not raise

    def test_assert_list_of_lines_invalid(self):
        # Invalid input: one line is not a 2x2 array
        lines = np.asarray([[0, 0], [1, 1], [2, 2]])
        with pytest.raises(ValueError):
            assert_list_of_lines(lines)
