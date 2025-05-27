"""
Unit Tests for the generic image methods
"""

import pytest
import numpy as np

from src.geometry import Polygon, Segment, Rectangle, LinearSpline
from src.image import Image, PolygonsRender, SegmentsRender, LinearSplinesRender


class TestImageIOU:

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


class TestImageScoreDistanceFromCenter:

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


class TestImageScoreContainsBase:

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


class TestImageScoreContainsSegments:

    def test_score_contains_segment_one(self):
        shape = (20, 20)
        img = Image.from_fillvalue(shape=shape, value=255)
        segment = Segment(points=[[10, 5], [15, 15]])
        img.draw_segments(
            segments=[segment],
            render=SegmentsRender(default_color=(0, 0, 0), thickness=3),
        )
        assert (
            img.score_contains_segments(segments=[segment], dilate_iterations=1)[0]
            == 1.0
        )

    # def test_score_contains_segment_vertical(self):
    # TODO solve error with vertical segments
    # shape = (20, 20)
    # img = Image.from_fillvalue(shape=shape, value=255)
    # segment = Segment(points=[[10, 5], [10, 15]])
    # img.draw_segments(
    #     segments=[segment], render=SegmentsRender(default_color=(0, 0, 0), thickness=3)
    # )
    # assert img.score_contains_segments(segments=[segment], dilate_iterations=1)[0] == 1.0

    def test_score_contains_segment_horizontal(self):
        shape = (20, 20)
        img = Image.from_fillvalue(shape=shape, value=255)
        segment = Segment(points=[[5, 5], [15, 5]])
        img.draw_segments(
            segments=[segment],
            render=SegmentsRender(default_color=(0, 0, 0), thickness=3),
        )
        assert (
            img.score_contains_segments(segments=[segment], dilate_iterations=1)[0]
            == 1.0
        )

    def test_score_contains_segments_zero(self):
        shape = (20, 20)
        img = Image.from_fillvalue(shape=shape, value=255)
        segment = Segment(points=[[2, 4], [17, 4]])
        other_segment = Segment(points=[[2, 15], [17, 15]])
        img.draw_segments(
            segments=[segment],
            render=SegmentsRender(default_color=(0, 0, 0), thickness=1),
        )
        scores = img.score_contains_segments(segments=[other_segment])
        assert scores[0] == 0.0

    def test_score_contains_segments_partial(self):
        shape = (20, 20)
        img = Image.from_fillvalue(shape=shape, value=255)
        partial_segment = Segment(points=[[2, 10], [17, 10]])
        segment = Segment(points=[[2, 10], [10, 10]])
        img.draw_segments(
            segments=[segment],
            render=SegmentsRender(default_color=(0, 0, 0), thickness=1),
        )
        scores = img.score_contains_segments(segments=[partial_segment])
        assert 0 < scores[0] < 1.0

    def test_score_contains_segments_multiple(self):
        shape = (20, 20)
        img = Image.from_fillvalue(shape=shape, value=255)
        segment1 = Segment(points=[[2, 10], [17, 10]])
        segment2 = Segment(points=[[2, 15], [17, 15]])
        img.draw_segments(
            segments=[segment1, segment2],
            render=SegmentsRender(default_color=(0, 0, 0), thickness=1),
        )
        scores = img.score_contains_segments(
            segments=[segment1, segment2], dilate_iterations=2
        )
        assert scores[0] == 1.0
        assert scores[1] == 1.0

    def test_score_contains_segments_with_dilation(self):
        shape = (20, 20)
        img = Image.from_fillvalue(shape=shape, value=255)
        segment = Segment(points=[[2, 10], [17, 10]])
        close_segment = Segment(points=[[2, 11], [17, 11]])
        img.draw_segments(
            segments=[segment],
            render=SegmentsRender(default_color=(0, 0, 0), thickness=1),
        )
        scores = img.score_contains_segments(
            segments=[close_segment], dilate_kernel=(3, 3), dilate_iterations=3
        )
        assert scores[0] == 1.0


class TestImageScoreContainsLinearSplines:

    def test_score_contains_linear_splines_one(self):
        shape = (500, 500)
        img = Image.from_fillvalue(shape=shape, value=255)
        splines = [
            LinearSpline(
                points=[
                    [10, 10],
                    [10, shape[0] - 10],
                    [shape[1] - 10, shape[0] - 10],
                    [shape[1] - 10, 10],
                ]
            )
        ]
        img.draw_splines(
            splines=splines, render=LinearSplinesRender(default_color=(0, 0, 0))
        )
        scores = img.score_contains_linear_splines(splines=splines)
        assert scores[0] == 1.0

    def test_score_contains_linear_splines_zero(self):
        shape = (50, 50)
        img = Image.from_fillvalue(shape=shape, value=255)
        other_spline = LinearSpline(
            points=[
                [30, 30],
                [60, 70],
                [70, 70],
                [70, 60],
            ]
        )
        scores = img.score_contains_linear_splines(splines=[other_spline])
        assert scores[0] == 0.0

    def test_score_contains_linear_splines_partial_v1(self):
        shape = (50, 50)
        img = Image.from_fillvalue(shape=shape, value=255)
        spline = LinearSpline(
            points=[
                [10, 10],
                [10, 30],
                [30, 30],
                [30, 10],
            ]
        )
        img.draw_splines(
            splines=[spline],
            render=LinearSplinesRender(default_color=(0, 0, 0), thickness=1),
        )
        partial_spline = LinearSpline(
            points=[
                [10, 10],
                [10, 25],
                [25, 25],
                [25, 15],
            ]
        )
        scores = img.score_contains_linear_splines(splines=[partial_spline])
        assert 0 < scores[0] < 1.0

    @pytest.mark.parametrize(
        "partial_spline, expected_score",
        [
            (
                LinearSpline(
                    points=[[100, 100], [100, 200], [200, 200], [200, 100], [300, 100]]
                ),
                0.25,
            ),
            (
                LinearSpline(points=[[100, 100], [100, 200], [200, 200], [200, 100]]),
                0.33,
            ),
            (LinearSpline(points=[[100, 100], [100, 200], [200, 200]]), 0.5),
        ],
    )
    def test_score_contains_linear_splines_partial_quarter(
        self, partial_spline, expected_score
    ):
        shape = (500, 500)
        img = Image.from_fillvalue(shape=shape, value=255)
        spline = LinearSpline(
            points=[
                [100, 100],
                [100, 300],
                [300, 300],
                [300, 150],
            ]
        )
        img.draw_splines(
            splines=[spline],
            render=LinearSplinesRender(default_color=(0, 0, 0), thickness=5),
        )
        scores = img.score_contains_linear_splines(
            splines=[partial_spline], dilate_iterations=0
        )
        assert abs(scores[0] - expected_score) < 0.1

    def test_score_contains_linear_splines_multiple(self):
        shape = (500, 500)
        img = Image.from_fillvalue(shape=shape, value=255)
        splines = [
            LinearSpline(
                points=[
                    [10, 10],
                    [10, shape[0] - 10],
                    [shape[1] - 10, shape[0] - 10],
                    [shape[1] - 10, 10],
                ]
            ),
            LinearSpline(
                points=[
                    [100, 100],
                    [100, 300],
                    [300, 300],
                    [300, 100],
                ]
            ),
        ]
        img.draw_splines(
            splines=splines, render=LinearSplinesRender(default_color=(0, 0, 0))
        )
        scores = img.score_contains_linear_splines(splines=splines)
        assert scores[0] == 1.0
        assert scores[1] == 1.0


class TestImageScoreContainsPolygons:

    def test_score_contains_polygons_one(self):
        shape = (50, 50)
        img = Image.from_fillvalue(shape=shape, value=255)
        polygon = Polygon(
            points=[
                [10, 10],
                [10, shape[0] - 10],
                [shape[1] - 10, shape[0] - 10],
                [shape[1] - 10, 10],
            ]
        )
        img.draw_polygons(
            polygons=[polygon],
            render=PolygonsRender(default_color=(0, 0, 0), thickness=1),
        )
        scores = img.score_contains_polygons(polygons=[polygon])
        assert scores[0] == 1.0

    def test_score_contains_polygons_zero(self):
        shape = (50, 50)
        img = Image.from_fillvalue(shape=shape, value=255)
        polygon = Polygon(
            points=[
                [30, 30],
                [60, 70],
                [70, 70],
                [70, 60],
            ]
        )
        scores = img.score_contains_polygons(polygons=[polygon])
        assert scores[0] == 0.0

    @pytest.mark.parametrize(
        "partial_polygon, expected_score",
        [
            (
                Polygon(
                    points=[[100, 100], [100, 200], [200, 200], [200, 100], [300, 100]]
                ),
                0.25,
            ),
            (
                Polygon(points=[[100, 100], [100, 200], [200, 200], [200, 100]]),
                0.33,
            ),
            (Polygon(points=[[100, 100], [100, 200], [200, 200]]), 0.5),
        ],
    )
    def test_score_contains_linear_polygons_partial_quarter(
        self, partial_polygon, expected_score
    ):
        shape = (500, 500)
        img = Image.from_fillvalue(shape=shape, value=255)
        polygon = Polygon(
            points=[
                [100, 100],
                [100, 300],
                [300, 300],
                [300, 150],
            ]
        )
        img.draw_polygons(
            polygons=[polygon],
            render=PolygonsRender(default_color=(0, 0, 0), thickness=5),
        )
        scores = img.score_contains_linear_splines(
            splines=[partial_polygon], dilate_iterations=0
        )
        assert abs(scores[0] - expected_score) < 0.1

    def test_score_contains_polygons_multiple(self):
        shape = (100, 100)
        img = Image.from_fillvalue(shape=shape, value=255)
        polygons = [
            Polygon(
                points=[
                    [10, 10],
                    [10, 50],
                    [50, 50],
                    [50, 10],
                ]
            ),
            Polygon(
                points=[
                    [60, 60],
                    [60, 90],
                    [90, 90],
                    [90, 60],
                ]
            ),
        ]
        img.draw_polygons(
            polygons=polygons,
            render=PolygonsRender(default_color=(0, 0, 0), thickness=1),
        )
        scores = img.score_contains_polygons(polygons=polygons)
        assert scores[0] == 1.0
        assert scores[1] == 1.0

    def test_score_contains_polygons_with_dilation(self):
        shape = (50, 50)
        img = Image.from_fillvalue(shape=shape, value=255)
        polygon = Polygon(
            points=[
                [10, 10],
                [10, 30],
                [30, 30],
                [30, 10],
            ]
        )
        img.draw_polygons(
            polygons=[polygon],
            render=PolygonsRender(default_color=(0, 0, 0), thickness=1),
        )
        smaller_polygon = Polygon(
            points=[
                [12, 12],
                [12, 28],
                [28, 28],
                [28, 12],
            ]
        )
        scores = img.score_contains_polygons(
            polygons=[smaller_polygon], dilate_kernel=(3, 3), dilate_iterations=3
        )
        assert scores[0] == 1.0


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
