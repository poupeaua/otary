"""
Init file for image module to facilitate importation
"""

# ruff: noqa: F401
from src.image.render import (
    PointsRender,
    ContoursRender,
    SegmentsRender,
    OcrSingleOutputRender,
)
from src.image.image import Image
