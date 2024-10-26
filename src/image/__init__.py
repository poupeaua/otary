"""
Init file for image module to facilitate importation
"""

# ruff: noqa: F401
from src.image.utils.render import (
    PointsRender,
    ContoursRender,
    SegmentsRender,
    OcrSingleOutputRender,
)
from src.image.image import Image
from src.image.utils.tools import interpolate_color
