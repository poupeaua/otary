"""
Contour class to handle complexity with contour calculation
"""

import cv2
import copy
import numpy as np
from src.utils.geometry import assert_is_array_of_lines

class Contour:
    
    def __init__(self, lines: np.array) -> None:
        self._check_lines_form_contour(lines)
        self.lines = copy.deepcopy(lines)
        self.area = cv2.contourArea(self.points)
        self.perimeter = cv2.arcLength(self.points, True)
        pass
    
    def _check_lines_form_contour(lines: np.array):
        #TODO
        pass
    
    @classmethod
    def from_points(cls, points: np.array):
        #TODO
        pass
    
    @classmethod
    def from_points_dedup(cls, points: np.array):
        #TODO
        pass
    
    @classmethod
    def from_lines_approx(cls, lines: np.array):
        #TODO
        pass
    
    def as_points(self):
        """Return contour as a array of successive points
        """
        #TODO
        pass
    
    def __repr__(self) -> str:
        """Return the contour as a array of successive lines

        Returns:
            str: _description_
        """
        return self.points
 