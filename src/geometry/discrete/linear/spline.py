"""
Curve class useful to describe any kind of curves
"""

from numpy.typing import NDArray

from src.geometry.discrete.linear.entity import LinearEntity


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

    @property
    def center_within(self) -> NDArray:
        """Returns the center point that is within the linear spline.
        This means that this points necessarily belongs to the linear spline.

        This can be useful when the centroid is not a good representation of what
        is needed as 'center'.

        Returns:
            NDArray: point of shape (1, 2)
        """
        return self.find_interpolated_point(pct_dist=0.5)

    def find_interpolated_point(self, pct_dist: float):
        """Return a point along the curve at a relative distance pct_dist ∈ [0, 1]

        Parameters:
            pct_dist (float): Value in [0, 1], 0 returns start, 1 returns end.
                Any value in [0, 1] returns a point between start and end that is
                pct_dist along the path.

        Returns:
            NDArray: Interpolated point [x, y]
        """
        if not (0 <= pct_dist <= 1):
            raise ValueError("pct_dist must be in [0, 1]")

        if self.length == 0 or pct_dist == 0:
            return self[0]
        if pct_dist == 1:
            return self[-1]

        # Walk along the path to find the point at pct_dist * total_dist
        target_dist = pct_dist * self.length
        accumulated = 0
        for i in range(len(self.edges)):
            cur_edge_length = self.lengths[i]
            if accumulated + cur_edge_length >= target_dist:
                remain = target_dist - accumulated
                direction = self[i + 1] - self[i]
                unit_dir = direction / cur_edge_length
                return self[i] + remain * unit_dir
            accumulated += cur_edge_length

        # Fallback
        return self[-1]
