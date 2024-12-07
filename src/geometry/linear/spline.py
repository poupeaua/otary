"""
Curve class useful to describe any kind of curves
"""

from src.geometry.linear.entity import LinearEntity


class LinearSpline(LinearEntity):
    """Curve class"""

    def curvature(self) -> float:
        # TODO
        return NotImplementedError
