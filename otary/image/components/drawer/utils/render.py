"""
Drawing Render used to makes easy drawings
"""

from dataclasses import dataclass, field
from abc import ABC

import cv2

from otary.image.components.drawer.utils.tools import is_color_tuple

DEFAULT_RENDER_THICKNESS = 3
DEFAULT_RENDER_COLOR = (0, 0, 255)


@dataclass(kw_only=True)
class Render(ABC):
    """Render class used to facilitate the rendering of objects when drawing them"""

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
class GeometryRender(Render, ABC):
    """Base class for the rendering of GeometryEntity objects"""


@dataclass
class PointsRender(GeometryRender):
    """Render for Point objects"""

    radius: int = 1


@dataclass
class EllipseRender(GeometryRender):
    """Render for Ellipse objects"""

    is_draw_center_point_enabled: bool = False


@dataclass
class CirclesRender(EllipseRender):
    """Render for Circle objects"""


@dataclass
class SegmentsRender(GeometryRender):
    """Render for Segment objects"""

    as_vectors: bool = False
    tip_length: int = 20


@dataclass
class LinearSplinesRender(SegmentsRender):
    """Render for Linear Splines objects"""

    pct_ix_head: float = 0.25


@dataclass
class PolygonsRender(SegmentsRender):
    """Render for Polygon objects. It inherits from SegmentsRender because
    Polygons are drawn as a succession of drawn segments."""

    is_filled: bool = False


@dataclass
class OcrSingleOutputRender(Render):
    """Render for OcrSingleOutput objects"""
