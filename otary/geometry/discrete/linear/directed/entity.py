"""
Module to define Linear Directed Geometric Entity
"""

from abc import abstractmethod

from metpy.calc import angle_to_direction
import numpy as np
from numpy.typing import NDArray

from otary.geometry.discrete.linear.entity import LinearEntity


class DirectedLinearEntity(LinearEntity):
    """DirectedLinearEntity class"""

    @property
    @abstractmethod
    def cardinal_degree(self) -> float:
        """Return the cardinal degree of the Directed Linear Entity"""

    @property
    def head(self) -> NDArray:
        """Return the head point at the end-extremity of the arrow directed object.

        Returns:
            NDArray: the head point
        """
        return self.asarray[-1]

    @property
    def tail(self) -> NDArray:
        """Return the tail point at the start-extremity of the arrow directed object.

        Returns:
            NDArray: the tail point
        """
        return self.asarray[0]

    @property
    def origin(self) -> NDArray:
        """Representation shifted to the origin (0,0)
        It is the same entity but with the tail point at (0,0) and the other
        points shifted accordingly.
        """
        return self.asarray - self.tail

    @property
    def cv2_space_coords(self) -> NDArray:
        """Inverted coordinates in the cv2 space

        Returns:
            NDArray: with inverted coordinates
        """
        return np.roll(self.points, shift=1, axis=1)

    @property
    def is_x_first_pt_gt_x_last_pt(self) -> bool:
        """Whether the x coordinate of the first point is greater than the x
        coordinate of the second point that forms the directed linear entity

        Returns:
            bool: if x0 > x1 returns True, else False
        """
        return bool(self.asarray[0][0] > self.asarray[-1][0])

    @property
    def is_y_first_pt_gt_y_last_pt(self) -> bool:
        """Whether the y coordinate of the first point is greater than the y
        coordinate of the second point that forms the directed linear entity

        Returns:
            bool: if y0 > y1 returns True, else False
        """
        return bool(self.asarray[0][1] > self.asarray[-1][1])

    def cardinal_direction(self, full: bool = False, level: int = 2) -> str:
        """Cardinal direction

        Args:
            full (bool, optional): True returns full text (South), False returns
                abbreviated text (S). Defaults to False.
            level (int, optional): Level of detail (3 = N/NNE/NE/ENE/E...
                2 = N/NE/E/SE... 1 = N/E/S/W). Defaults to 2.

        Returns:
            str: _description_
        """
        return angle_to_direction(
            input_angle=self.cardinal_degree, full=full, level=level
        )
