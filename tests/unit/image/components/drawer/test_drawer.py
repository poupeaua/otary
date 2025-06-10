"""
Unit Tests for the drawer image methods
"""

import numpy as np

from src.cv.ocr.output.ocr_single_output import OcrSingleOutput
from src.geometry import Polygon, Rectangle, Circle, LinearSpline
from src.image import Image, SegmentsRender, CirclesRender, LinearSplinesRender


class TestDrawerPointsImage:

    def test_draw_points(self):
        points = np.array([[0, 0], [1, 1], [2, 3]])
        Image.from_fillvalue(shape=(5, 5, 3), value=0).draw_points(points=points)


class TestDrawerSegmentsImage:

    def test_draw_segments(self):
        segments = np.array([[[0, 0], [1, 1]]])
        Image.from_fillvalue(shape=(5, 5, 3), value=0).draw_segments(segments=segments)

    def test_draw_segments_arrowed(self):
        segments = np.array([[[0, 0], [1, 1]]])
        Image.from_fillvalue(shape=(5, 5, 3), value=0).draw_segments(
            segments=segments, render=SegmentsRender(as_vectors=True)
        )


class TestDrawerPolygonsImage:

    def test_draw_polygons(self):
        points = np.array([[0, 0], [1, 1], [2, 3]])
        cnt = Polygon(points=points)
        Image.from_fillvalue(shape=(5, 5, 3), value=0).draw_polygons(polygons=[cnt])


class TestDrawerOcrImage:

    def test_draw_ocrso(self):
        ocrso = OcrSingleOutput(
            bbox=Rectangle([[0, 0], [0, 1], [1, 1], [1, 0]]),
            text="test_text",
            confidence=0.9,
        )
        Image.from_fillvalue(shape=(5, 5, 3), value=0).draw_ocr_outputs(
            ocr_outputs=[ocrso]
        )

    def test_draw_ocrso_empty(self):
        ocrso = OcrSingleOutput(
            bbox=None,
            text=None,
            confidence=None,
        )
        Image.from_fillvalue(shape=(5, 5, 3), value=0).draw_ocr_outputs(
            ocr_outputs=[ocrso]
        )


class TestDrawerCirclesImage:

    def test_draw_circles(self):

        circles = [
            Circle(center=np.array([2, 2]), radius=1),
            Circle(center=np.array([4, 4]), radius=2),
        ]
        Image.from_fillvalue(shape=(10, 10, 3), value=0).draw_circles(circles=circles)

    def test_draw_circles_with_render(self):

        circles = [
            Circle(center=np.array([3, 3]), radius=2),
            Circle(center=np.array([6, 6]), radius=3),
        ]
        render = CirclesRender(thickness=2, is_draw_center_point_enabled=True)
        Image.from_fillvalue(shape=(15, 15, 3), value=0).draw_circles(
            circles=circles, render=render
        )


class TestDrawerSplinesImage:

    def test_draw_splines(self):
        splines = [
            LinearSpline(points=np.array([[0, 0], [1, 1], [2, 2]])),
            LinearSpline(points=np.array([[3, 3], [4, 4], [5, 5]])),
        ]
        Image.from_fillvalue(shape=(10, 10, 3), value=0).draw_splines(splines=splines)

    def test_draw_splines_with_render(self):
        splines = [
            LinearSpline(points=np.array([[0, 0], [1, 1], [2, 2]])),
            LinearSpline(points=np.array([[3, 3], [4, 4], [5, 5]])),
        ]
        render = LinearSplinesRender(thickness=2, as_vectors=True)
        Image.from_fillvalue(shape=(10, 10, 3), value=0).draw_splines(
            splines=splines, render=render
        )
