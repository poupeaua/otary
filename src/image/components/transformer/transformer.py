"""
Image Trasnformation module. it only contains advanced image transformation methods.
"""

from __future__ import annotations

from src.image.base import BaseImage
from src.image.components.transformer.components import (
    BinarizerImage,
    CropperImage,
    GeometrizerImage,
    MorphologyzerImage,
)


class TransformerImage:
    """Transformer images utility class"""

    def __init__(self, base: BaseImage):
        self._base = base
        self.binarizer = BinarizerImage(base=self._base)
        self.cropper = CropperImage(base=self._base)
        self.geometrizer = GeometrizerImage(base=self._base)
        self.morphologyzer = MorphologyzerImage(base=self._base)
