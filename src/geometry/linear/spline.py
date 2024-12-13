"""
Curve class useful to describe any kind of curves
"""

from src.geometry.linear.entity import LinearEntity


class LinearSpline(LinearEntity):
    """Curve class"""

    @property
    def curvature(self) -> float:
        """Get the curvature of the linear spline

        Returns:
            float: curvature value
        """
        # TODO
        raise NotImplementedError
