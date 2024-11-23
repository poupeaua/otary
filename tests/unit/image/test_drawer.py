"""
Unit Tests for the drawer image methods
"""

import numpy as np

from src.core.dataclass.ocrsingleoutput import OcrSingleOutput
from src.geometry import Contour, Rectangle
from src.image import Image, SegmentsRender


class TestDrawerImage:

    def test_draw_circles(self):
        points = np.array([[0, 0], [1, 1], [2, 3]])
        Image.from_fillvalue(shape=(5, 5, 3), value=0).draw_circles(points=points)

    def test_draw_segments(self):
        segments = np.array([[[0, 0], [1, 1]]])
        Image.from_fillvalue(shape=(5, 5, 3), value=0).draw_segments(segments=segments)

    def test_draw_segments_arrowed(self):
        segments = np.array([[[0, 0], [1, 1]]])
        Image.from_fillvalue(shape=(5, 5, 3), value=0).draw_segments(
            segments=segments, render=SegmentsRender(as_vectors=True)
        )

    def test_draw_contours(self):
        points = np.array([[0, 0], [1, 1], [2, 3]])
        cnt = Contour(points=points)
        Image.from_fillvalue(shape=(5, 5, 3), value=0).draw_contours(contours=[cnt])

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
