"""
Triangle class module
"""

from src.geometry import Contour


class Triangle(Contour):
    """Triangle class"""

    def __init__(self, points) -> None:
        assert len(points) == 3
        super().__init__(points)
