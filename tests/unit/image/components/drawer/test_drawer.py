"""
Unit Tests for the drawer image methods
"""

import numpy as np

from otary.utils.cv.ocrsingleoutput import OcrSingleOutput
from otary.geometry import Polygon, Rectangle, Circle, Ellipse, LinearSpline
from otary.image import (
    Image,
    SegmentsRender,
    CirclesRender,
    LinearSplinesRender,
    EllipsesRender,
    PolygonsRender,
)


class TestDrawerColors:

    def test_draw_colors_as_tuple(self):
        points = np.array([[0, 0], [1, 1], [2, 3]])
        cnt = Polygon(points=points)
        Image.from_fillvalue(shape=(5, 5, 3), value=0).draw_polygons(
            polygons=[cnt], render=PolygonsRender(colors=[(0, 0, 0)])
        )

    def test_draw_colors_as_tuple_bad(self):
        points = np.array([[0, 0], [1, 1], [2, 3]])
        cnt = Polygon(points=points)
        Image.from_fillvalue(shape=(5, 5, 3), value=0).draw_polygons(
            polygons=[cnt], render=PolygonsRender(colors=[(0, -1, 300)])
        )

    def test_draw_colors_as_str(self):
        points = np.array([[0, 0], [1, 1], [2, 3]])
        cnt = Polygon(points=points)
        Image.from_fillvalue(shape=(5, 5, 3), value=0).draw_polygons(
            polygons=[cnt], render=PolygonsRender(colors=["blue"])
        )

    def test_draw_colors_as_str_bad(self):
        points = np.array([[0, 0], [1, 1], [2, 3]])
        cnt = Polygon(points=points)
        Image.from_fillvalue(shape=(5, 5, 3), value=0).draw_polygons(
            polygons=[cnt], render=PolygonsRender(colors=["pi$?7_="])
        )

    def test_draw_default_color_as_str(self):
        points = np.array([[0, 0], [1, 1], [2, 3]])
        cnt = Polygon(points=points)
        Image.from_fillvalue(shape=(5, 5, 3), value=0).draw_polygons(
            polygons=[cnt], render=PolygonsRender(default_color=["blue"])
        )

    def test_draw_default_color_as_str_bad(self):
        points = np.array([[0, 0], [1, 1], [2, 3]])
        cnt = Polygon(points=points)
        Image.from_fillvalue(shape=(5, 5, 3), value=0).draw_polygons(
            polygons=[cnt], render=PolygonsRender(default_color=["pi$?7_="])
        )


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

    def test_draw_polygons_with_filled_render(self):
        points = np.array([[0, 0], [1, 1], [2, 3]])
        cnt = Polygon(points=points)
        render = PolygonsRender(is_filled=True)
        Image.from_fillvalue(shape=(5, 5, 3), value=0).draw_polygons(
            polygons=[cnt], render=render
        )


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


class TestDrawerEllipsesImage:

    def test_draw_ellipses_basic(self):
        ellipses = [
            Ellipse(
                foci1=np.array([5, 5]), foci2=np.array([10, 10]), semi_major_axis=10
            ),
            Ellipse(
                foci1=np.array([10, 10]), foci2=np.array([20, 15]), semi_major_axis=15
            ),
        ]
        img = Image.from_fillvalue(shape=(20, 20, 3), value=0)
        img.draw_ellipses(ellipses=ellipses)

    def test_draw_ellipses_with_render_options(self):
        ellipses = [
            Ellipse(
                foci1=np.array([5, 5]), foci2=np.array([10, 10]), semi_major_axis=10
            )
        ]
        render = EllipsesRender(
            thickness=2,
            is_filled=True,
            is_draw_center_point_enabled=True,
            is_draw_focis_enabled=True,
        )
        img = Image.from_fillvalue(shape=(20, 20, 3), value=0)
        img.draw_ellipses(ellipses=ellipses, render=render)

    def test_draw_ellipses_empty(self):
        img = Image.from_fillvalue(shape=(10, 10, 3), value=0)
        img.draw_ellipses(ellipses=[])
