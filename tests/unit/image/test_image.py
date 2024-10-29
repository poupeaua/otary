"""
Unit Tests for the generic image methods
"""

import pytest
import numpy as np

from src.geometry import Contour, Segment
from src.image import Image, ContoursRender, SegmentsRender


class TestImageGlobalMethods:

    def test_iou(self):
        img0 = Image.from_fillvalue(shape=(5, 5), value=255)
        for x in range(2, 4):
            for y in range(2, 4):
                img0.asarray[x, y] = 0
        img1 = img0.copy().rotate(angle=180)
        assert img0.iou(img1) == 1 / 7

    def test_iou_zero(self):
        img0 = Image.from_fillvalue(shape=(5, 5), value=255)
        for x in range(3, 5):
            for y in range(3, 5):
                img0.asarray[x, y] = 0
        img1 = img0.copy().rotate(angle=180)
        assert img0.iou(img1) == 0

    def test_iou_one(self):
        img0 = Image.from_fillvalue(shape=(5, 5), value=255)
        for x in range(3, 5):
            for y in range(3, 5):
                img0.asarray[x, y] = 0
        img1 = img0.copy()
        assert img0.iou(img1) == 1


class TestImageScoreMethods:

    def test_score_contains(self):
        img0 = Image.from_fillvalue(shape=(5, 5), value=255)
        for x in range(2, 4):
            for y in range(2, 4):
                img0.asarray[x, y] = 0
        img1 = img0.copy().rotate(angle=180)
        assert img0.score_contains(img1) == 1 / 4

    def test_score_contains_zero(self):
        img0 = Image.from_fillvalue(shape=(5, 5), value=255)
        for x in range(3, 5):
            for y in range(3, 5):
                img0.asarray[x, y] = 0
        img1 = img0.copy().rotate(angle=180)
        assert img0.score_contains(img1) == 0

    def test_score_contains_one(self):
        img0 = Image.from_fillvalue(shape=(5, 5), value=255)
        for x in range(3, 5):
            for y in range(3, 5):
                img0.asarray[x, y] = 0
        img1 = img0.copy()
        assert img0.score_contains(img1) == 1

    def test_score_contains_contour_one(self):
        shape = (5, 5)
        img = Image.from_fillvalue(shape=shape, value=255)
        cnt = Contour(
            points=[[0, 0], [0, shape[0]], [shape[1], shape[0]], [shape[1], 0]]
        )
        img.draw_contours(
            contours=[cnt], render=ContoursRender(default_color=(0, 0, 0))
        )
        assert img.score_contains_contour(contour=cnt) == 1

    def test_score_contains_segment_one(self):
        shape = (5, 5)
        img = Image.from_fillvalue(shape=shape, value=255)
        segment = Segment(points=[[0, 0], [0, shape[0]]])
        img.draw_segments(
            segments=[segment], render=SegmentsRender(default_color=(0, 0, 0))
        )
        assert img.score_contains_segments(segments=[segment]) == 1

    def test_score_distance_from_center_error_method(self):
        with pytest.raises(ValueError):
            Image.from_fillvalue(shape=(5, 5), value=0).score_distance_from_center(
                point=[0, 0], method="method_not_exist"
            )

    def test_score_distance_from_center_linear1(self):
        shape = (101, 101)
        point = (np.array(list(shape)) - 1) / 2
        score = Image.from_fillvalue(shape=shape, value=0).score_distance_from_center(
            point=point, method="linear"
        )
        assert score == 1

    def test_score_distance_from_center_linear0(self):
        shape = (101, 101)
        point = np.array([0, 0])
        score = Image.from_fillvalue(shape=shape, value=0).score_distance_from_center(
            point=point, method="linear"
        )
        assert score == 0

    def test_score_distance_from_center_gaussian1(self):
        shape = (101, 101)
        point = (np.array(list(shape)) - 1) / 2
        score = Image.from_fillvalue(shape=shape, value=0).score_distance_from_center(
            point=point, method="gaussian"
        )
        assert score == 1

    def test_score_distance_from_center_gaussian0(self):
        shape = (101, 101)
        point = np.array([0, 0])
        score = Image.from_fillvalue(shape=shape, value=0).score_distance_from_center(
            point=point, method="gaussian"
        )
        assert round(score) == 0
