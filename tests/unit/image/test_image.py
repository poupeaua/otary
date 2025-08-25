"""
Unit Tests for the generic image methods
"""

import os

import pytest
import numpy as np

from otary.geometry import (
    Polygon,
    Segment,
    LinearSpline,
    Rectangle,
    AxisAlignedRectangle,
)
from otary.image import Image, PolygonsRender, SegmentsRender, LinearSplinesRender


class TestImageStr:

    def test_str(self):
        img = Image.from_fillvalue(shape=(5, 5), value=255)
        assert "Image(" in str(img)


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


class TestImageCropRectangle:
    def test_crop_rectangle_axis_aligned(self):
        img = Image.from_fillvalue(shape=(100, 100), value=255)
        rect = Rectangle(points=[[10, 10], [10, 30], [30, 30], [30, 10]])
        cropped = img.crop_rectangle(rect)
        assert cropped.asarray.shape[0] == 20
        assert cropped.asarray.shape[1] == 20

    def test_crop_rectangle_rotated(self):
        img = Image.from_fillvalue(shape=(100, 100), value=255)
        # Rectangle rotated 45 degrees
        rect = Rectangle(points=[[50, 40], [60, 50], [50, 60], [40, 50]])
        cropped = img.crop_rectangle(rect)
        # The width and height should be equal for a square rotated 45 degrees
        assert cropped.asarray.shape[0] == cropped.asarray.shape[1]
        assert cropped.asarray.shape[0] > 0

    def test_crop_rectangle_with_different_topleft_ix(self):
        img = Image.from_fillvalue(shape=(100, 100), value=255)
        rect = Rectangle(points=[[10, 10], [10, 30], [30, 30], [30, 10]])
        cropped0 = img.crop_rectangle(rect, rect_topleft_ix=0)
        cropped1 = img.crop_rectangle(rect, rect_topleft_ix=1)
        assert cropped0.asarray.shape == cropped1.asarray.shape

    def test_crop_rectangle_out_of_bounds(self):
        img = Image.from_fillvalue(shape=(50, 50), value=255)
        rect = Rectangle(points=[[-10, -10], [-10, 10], [10, 10], [10, -10]])
        cropped = img.crop_rectangle(rect)
        assert cropped.asarray.shape[0] > 0
        assert cropped.asarray.shape[1] > 0

    def test_crop_rectangle_returns_image_instance(self):
        img = Image.from_fillvalue(shape=(20, 20), value=255)
        rect = Rectangle(points=[[5, 5], [5, 15], [15, 15], [15, 5]])
        cropped = img.crop_rectangle(rect)
        assert isinstance(cropped, Image)


class TestImageScoreContainsLinearEntities:

    def test_score_contains_linear_entities_segment_one(self):
        shape = (50, 50)
        img = Image.from_fillvalue(shape=shape, value=255)
        segment = Segment(points=[[10, 10], [40, 40]])
        img.draw_segments(
            segments=[segment],
            render=SegmentsRender(default_color=(0, 0, 0), thickness=7),
        )
        scores = img.score_contains_linear_entities([segment])
        assert scores[0] == 1.0

    def test_score_contains_linear_entities_spline_one(self):
        shape = (50, 50)
        img = Image.from_fillvalue(shape=shape, value=255)
        spline = LinearSpline(points=[[10, 10], [25, 40], [40, 10]])
        img.draw_splines(
            splines=[spline],
            render=LinearSplinesRender(default_color=(0, 0, 0), thickness=3),
        )
        scores = img.score_contains_linear_entities([spline])
        assert scores[0] == 1.0

    def test_score_contains_linear_entities_mixed(self):
        shape = (50, 50)
        img = Image.from_fillvalue(shape=shape, value=255)
        segment = Segment(points=[[5, 5], [45, 5]])
        spline = LinearSpline(points=[[10, 10], [25, 40], [40, 10]])
        img.draw_segments(
            segments=[segment],
            render=SegmentsRender(default_color=(0, 0, 0), thickness=10),
        )
        img.draw_splines(
            splines=[spline],
            render=LinearSplinesRender(default_color=(0, 0, 0), thickness=10),
        )
        scores = img.score_contains_linear_entities([segment, spline])
        assert scores[0] == 1.0
        assert scores[1] == 1.0

    def test_score_contains_linear_entities_partial(self):
        shape = (50, 50)
        img = Image.from_fillvalue(shape=shape, value=255)
        segment = Segment(points=[[5, 5], [25, 5]])
        img.draw_segments(
            segments=[segment],
            render=SegmentsRender(default_color=(0, 0, 0), thickness=2),
        )
        partial_segment = Segment(points=[[5, 5], [45, 5]])
        scores = img.score_contains_linear_entities([partial_segment])
        assert 0 < scores[0] < 1.0

    def test_score_contains_linear_entities_zero(self):
        shape = (50, 50)
        img = Image.from_fillvalue(shape=shape, value=255)
        segment = Segment(points=[[5, 5], [25, 5]])
        img.draw_segments(
            segments=[segment],
            render=SegmentsRender(default_color=(0, 0, 0), thickness=2),
        )
        other_segment = Segment(points=[[30, 30], [45, 45]])
        scores = img.score_contains_linear_entities([other_segment])
        assert scores[0] == 0.0

    def test_score_contains_linear_entities_invalid_type(self):
        shape = (50, 50)
        img = Image.from_fillvalue(shape=shape, value=255)

        class DummyEntity:
            pass

        dummy = DummyEntity()
        with pytest.raises(TypeError):
            img.score_contains_linear_entities([dummy])


class TestImageCropHQFromAABBAndPDF:

    def test_crop_hq_from_aabb_and_pdf(self):
        # Prepare a rectangle to crop (arbitrary values within a typical A4 page)
        bbox = AxisAlignedRectangle(points=[[50, 50], [100, 50], [100, 250], [50, 250]])
        factor_scale = bbox.get_height_from_topleft(0) / bbox.get_width_from_topleft(0)
        pdf_path = "tests/data/test.pdf"
        # Ensure the test PDF exists
        assert os.path.exists(pdf_path), f"Test PDF not found at {pdf_path}"

        # Call the method
        img = Image.from_pdf(pdf_path, page_nb=0, as_grayscale=True, resolution=400)
        cropped = img.crop_hq_from_aabb_and_pdf(
            bbox=bbox,
            pdf_filepath=pdf_path,
            page_nb=0,
            as_grayscale=True,
            resolution=800,
        )

        # Check that the result is an Image instance
        assert isinstance(cropped, Image)
        # Check that the cropped image is not empty
        assert cropped.asarray.size > 0

        assert factor_scale == pytest.approx(cropped.height / cropped.width, rel=0.05)
