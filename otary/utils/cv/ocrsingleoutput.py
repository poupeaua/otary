"""
OCR Single Output dataclass
"""

from typing import Optional

from otary.geometry.discrete.shape.rectangle import Rectangle
from dataclasses import dataclass

@dataclass
class OcrSingleOutput:
    bbox: Rectangle
    objectness: Optional[float] = None
    text: Optional[str] = None
    confidence: Optional[float] = None