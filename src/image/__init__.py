"""
Init file for image module to facilitate importation
"""

__all__ = [
    "CirclesRender",
    "PolygonsRender",
    "SegmentsRender",
    "OcrSingleOutputRender",
    "Image",
    "interpolate_color",
]

from src.image.utils.render import (
    CirclesRender,
    PolygonsRender,
    SegmentsRender,
    OcrSingleOutputRender,
)
from src.image.image import Image
from src.image.utils.tools import interpolate_color
