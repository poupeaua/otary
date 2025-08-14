"""
Tests for the AxisAlignedRectangle class
"""

import pymupdf

from otary.geometry import AxisAlignedRectangle

class TestRectanglePyMuRect:

    def test_as_pymupdf_rect_axis_aligned(self):
        # Create an axis-aligned rectangle
        rect = AxisAlignedRectangle.from_topleft(topleft=[0, 0], width=2, height=4)
        pymupdf_rect = rect.as_pymupdf_rect

        # Assert the pymupdf.Rect object has correct coordinates
        assert isinstance(pymupdf_rect, pymupdf.Rect)
        assert pymupdf_rect.x0 == 0
        assert pymupdf_rect.y0 == 0
        assert pymupdf_rect.x1 == 2
        assert pymupdf_rect.y1 == 4
