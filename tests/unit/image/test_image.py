"""
Unit Tests for the generic image methods
"""

import pytest
import numpy as np

from src.geometry import Polygon, Segment, Rectangle
from src.image import Image, PolygonsRender, SegmentsRender


class TestImageGlobalMethods:

    def test_iou_zero(self):
        img0 = Image.from_fillvalue(shape=(5, 5), value=255)
        for x in range(3, 5):
            for y in range(3, 5):
                img0.asarray[x, y] = 0
        img1 = img0.copy().rotate(angle=180, is_degree=True)
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
        img1 = img0.copy().rotate(angle=180, is_degree=True, reshape=False, fast=False)
        assert img0.score_contains(img1) == 1 / 4

    def test_score_contains_zero(self):
        img0 = Image.from_fillvalue(shape=(5, 5), value=255)
        for x in range(3, 5):
            for y in range(3, 5):
                img0.asarray[x, y] = 0
        img1 = img0.copy().rotate(angle=180, is_degree=True)
        assert img0.score_contains(img1) == 0

    def test_score_contains_one(self):
        img0 = Image.from_fillvalue(shape=(5, 5), value=255)
        for x in range(3, 5):
            for y in range(3, 5):
                img0.asarray[x, y] = 0
        img1 = img0.copy()
        assert img0.score_contains(img1) == 1

    def test_score_contains_contour_one(self):
        shape = (50, 50)
        img = Image.from_fillvalue(shape=shape, value=255)
        cnt = Polygon(
            points=[
                [0, 0],
                [0, shape[0] - 1],
                [shape[1] - 1, shape[0] - 1],
                [shape[1] - 1, 0],
            ]
        )
        img.draw_polygons(
            polygons=[cnt], render=PolygonsRender(default_color=(0, 0, 0), thickness=1)
        )
        assert img.score_contains_polygon(polygon=cnt) == 1

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


class TestImageRestrictRectInFrame:

    def test_restrict_rect_in_frame_within_bounds(self):
        img = Image.from_fillvalue(shape=(10, 10), value=255)
        rect = Rectangle.from_topleft_bottomright(
            topleft=np.asarray([2, 2]),
            bottomright=np.asarray([8, 8]),
            is_cast_int=True,
        )
        restricted_rect = img.restrict_rect_in_frame(rect)
        assert restricted_rect.xmin == 2
        assert restricted_rect.ymin == 2
        assert restricted_rect.xmax == 8
        assert restricted_rect.ymax == 8

    def test_restrict_rect_in_frame_outside_bounds(self):
        img = Image.from_fillvalue(shape=(10, 10), value=255)
        rect = Rectangle.from_topleft_bottomright(
            topleft=np.asarray([-5, -5]),
            bottomright=np.asarray([15, 15]),
            is_cast_int=True,
        )
        restricted_rect = img.restrict_rect_in_frame(rect)
        assert restricted_rect.xmin == 0
        assert restricted_rect.ymin == 0
        assert restricted_rect.xmax == 10
        assert restricted_rect.ymax == 10

    def test_restrict_rect_in_frame_partial_outside_bounds(self):
        img = Image.from_fillvalue(shape=(10, 10), value=255)
        rect = Rectangle.from_topleft_bottomright(
            topleft=np.asarray([5, 5]),
            bottomright=np.asarray([15, 15]),
            is_cast_int=True,
        )
        restricted_rect = img.restrict_rect_in_frame(rect)
        assert restricted_rect.xmin == 5
        assert restricted_rect.ymin == 5
        assert restricted_rect.xmax == 10
        assert restricted_rect.ymax == 10

    def test_restrict_rect_in_frame_exact_bounds(self):
        img = Image.from_fillvalue(shape=(10, 10), value=255)
        rect = Rectangle.from_topleft_bottomright(
            topleft=np.asarray([0, 0]),
            bottomright=np.asarray([10, 10]),
            is_cast_int=True,
        )
        restricted_rect = img.restrict_rect_in_frame(rect)
        assert restricted_rect.xmin == 0
        assert restricted_rect.ymin == 0
        assert restricted_rect.xmax == 10
        assert restricted_rect.ymax == 10
