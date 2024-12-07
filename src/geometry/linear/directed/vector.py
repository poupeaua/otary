"""
Vectors class they are like segments, but with a given direction
"""

import numpy as np

from src.geometry import Segment, DirectedLinearEntity


class Vector(Segment, DirectedLinearEntity):
    """Vector class to manipulate vector which can be seen as Segment with direction"""

    @property
    def cardinal_degree(self) -> float:
        """Returns the cardinal degree of the vector in the cv2 space.
        We consider the top of the image to point toward the north as default and thus
        represent the cardinal degree value 0 mod 360.

        Returns:
            float: cardinal degree
        """
        angle = self.slope_angle(degree=True, is_cv2=True)

        # if angle is negative
        is_neg_sign_angle = bool(np.sign(angle) - 1)
        if is_neg_sign_angle:
            angle = 90 + np.abs(angle)
        else:
            angle = 90 - angle

        # if vector points towards west
        if self.is_x_first_pt_gt_x_last_pt:
            angle += 180

        cardinal_degree = np.mod(360 + angle, 360)  # avoid negative value case
        return cardinal_degree
