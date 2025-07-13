"""
Test Point function
"""

import numpy as np
import pytest
from otary.geometry.discrete.point import Point


class TestPointSetArray:
    def test_asarray_getter_returns_correct_value(self):
        arr = np.array([1.0, 2.0])
        p = Point(arr)
        np.testing.assert_array_equal(p.asarray, np.array([[1.0, 2.0]]))

    def test_asarray_setter_with_flat_array(self):
        p = Point(np.array([0.0, 0.0]))
        new_val = np.array([3.0, 4.0])
        p.asarray = new_val
        print(p.asarray)
        np.testing.assert_array_equal(p.asarray, new_val.reshape((1, 2)))

    def test_asarray_setter_with_2d_array(self):
        p = Point(np.array([0.0, 0.0]))
        new_val = np.array([[5.0, 6.0]])
        p.asarray = new_val
        np.testing.assert_array_equal(p.asarray, new_val)

    def test_asarray_setter_raises_on_invalid_shape(self):
        p = Point(np.array([0.0, 0.0]))
        with pytest.raises(ValueError):
            p.asarray = np.array([[1.0, 2.0], [3.0, 4.0]])

    def test_asarray_setter_casts_to_numpy_array(self):
        p = Point(np.array([0.0, 0.0]))
        p.asarray = [7.0, 8.0]
        np.testing.assert_array_equal(p.asarray, np.array([[7.0, 8.0]]))
