"""
Drawing Render used to makes easy drawings
"""

from typing import Any
from dataclasses import dataclass, field
from abc import ABC

import cv2
import numpy as np

DEFAULT_RENDER_THICKNESS = 3
DEFAULT_RENDER_COLOR = (0, 0, 255)


def is_color_tuple(color: Any) -> bool:
    """Identify if the input color parameter is in the expected format for a color

    Args:
        color (tuple): an expected python object to define a color

    Returns:
        bool: True if the input is a good color, False otherwise
    """
    cond = bool(
        isinstance(color, tuple)
        and len(color) == 3
        and np.all([isinstance(c, int) for c in color])
    )
    return cond


@dataclass(kw_only=True)
class DrawingRender(ABC):
    thickness: int = DEFAULT_RENDER_THICKNESS
    line_type: int = cv2.LINE_AA
    default_color: tuple[int, int, int] = DEFAULT_RENDER_COLOR
    colors: list[tuple[int, int, int]] = field(default_factory=list)

    def adjust_colors_length(self, n: int) -> None:
        """Correct the color parameter in case the objects has not the same length

        Args:
            n (int): number of objects to expect
        """
        if len(self.colors) > n:
            self.colors = self.colors[:n]
        elif len(self.colors) < n:
            n_missing = n - len(self.colors)
            self.colors = self.colors + [self.default_color for _ in range(n_missing)]

    def __post_init__(self):
        """DrawingRender post-initialization method"""
        # check that the colors parameter is conform
        for i, color in enumerate(self.colors):
            if not is_color_tuple(color):
                self.colors[i] = self.default_color


@dataclass
class GeometryRender(DrawingRender, ABC):
    pass


@dataclass
class PointsRender(GeometryRender):
    radius: int = 3


@dataclass
class SegmentsRender(GeometryRender):
    as_vectors: bool = False
    tip_length: int = 20


@dataclass
class ContoursRender(SegmentsRender):
    pass


@dataclass
class OcrSingleOutputRender(DrawingRender):
    pass
