"""
Test GeometryEntity file
"""

import pytest
import numpy as np

from src.geometry import Segment


class TestEntityShift:
    def test_shift_segment(self):
        seg = Segment([[1, 1], [2, 2]])
        shift_vector = [0, -2]
        assert np.array_equal(
            a1=seg.shift(vector=shift_vector).asarray, a2=np.array([[1, -1], [2, 0]])
        )

    def test_shift_segment_far_vector(self):
        seg = Segment([[1, 1], [2, 2]])
        shift_vector = [[10, 10], [10, 8]]
        assert np.array_equal(
            seg.shift(vector=shift_vector).asarray, np.array([[1, -1], [2, 0]])
        )

    def test_shift_segment_error(self):
        seg = Segment([[1, 1], [2, 2]])
        shift_vector = [[10, 10], [2, 2], [8, 9]]
        with pytest.raises(ValueError):
            seg.shift(vector=shift_vector)


class TestEntityRotate:
    def test_segment_rotate_pi_over_4(self):
        seg = Segment([[1, 1], [2, 2]])
        assert np.array_equal(
            np.round(seg.rotate(angle=np.pi / 4).asarray, 5),
            np.round([[1.5, 1.5 - np.sqrt(2) / 2], [1.5, 1.5 + np.sqrt(2) / 2]], 5),
        )

    def test_segment_rotate_pi_over_4_degree(self):
        seg = Segment([[1, 1], [2, 2]])
        assert np.array_equal(
            np.round(seg.rotate(angle=45, degree=True).asarray, 5),
            np.round([[1.5, 1.5 - np.sqrt(2) / 2], [1.5, 1.5 + np.sqrt(2) / 2]], 5),
        )

    def test_segment_rotate_pi_over_2(self):
        seg = Segment([[1, 1], [2, 2]])
        assert np.array_equal(
            seg.rotate(angle=np.pi / 2).asarray, np.array([[2, 1], [1, 2]])
        )

    def test_segment_rotate_pi(self):
        seg = Segment([[1, 1], [2, 2]])
        assert np.array_equal(
            np.round(seg.rotate(angle=np.pi).asarray, 5), np.round([[2, 2], [1, 1]], 5)
        )

    def test_segment_rotate_neg_pi_over_4(self):
        seg = Segment([[1, 1], [2, 2]])
        assert np.array_equal(
            np.round(seg.rotate(angle=-np.pi / 4).asarray, 5),
            np.round([[1.5 - np.sqrt(2) / 2, 1.5], [1.5 + np.sqrt(2) / 2, 1.5]], 5),
        )

    def test_segment_rotate_neg_pi_over_2(self):
        seg = Segment([[1, 1], [2, 2]])
        assert np.array_equal(
            seg.rotate(angle=-np.pi / 2).asarray, np.array([[1, 2], [2, 1]])
        )

    def test_segment_rotate_neg_pi(self):
        seg = Segment([[1, 1], [2, 2]])
        assert np.array_equal(
            np.round(seg.rotate(angle=-np.pi).asarray, 5), np.round([[2, 2], [1, 1]], 5)
        )

    def test_segment_rotate_around_image_center_pi(self):
        seg = Segment([[1, 1], [2, 2]])
        side = 100
        img = np.zeros(shape=(side, side))
        assert np.array_equal(
            seg.rotate_around_image_center(img=img, angle=np.pi).asarray,
            np.array([[side - 1, side - 1], [side - 2, side - 2]]),
        )

    def test_segment_rotate_around_image_center_pi_over_2(self):
        seg = Segment([[1, 1], [2, 2]])
        side = 100
        img = np.zeros(shape=(side, side))
        assert np.array_equal(
            seg.rotate_around_image_center(img=img, angle=np.pi / 2).asarray,
            np.array([[side - 1, 1], [side - 2, 2]]),
        )
