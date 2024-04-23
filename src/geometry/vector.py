"""
Vectors class they are like segments, but with a given direction
"""

import numpy as np
from metpy.calc import angle_to_direction
from src.geometry import Segment

class Vector(Segment):
    
    def __init__(self, points) -> None:
        super().__init__(points)
        
    @property
    def cv2_space_coords(self):
        return np.roll(self.points, 1)
        
    @property
    def is_x0_greater_than_x1(self):
        return bool(self.points[0][0] > self.points[1][0])
    
    @property
    def cardinal_degree(self) -> float:
        """Returns the cardinal degree of the vector in the cv2 space.
        We consider the top of the image to point toward the north as default and thus represent
        the cardinal degree value 0 mod 360.

        Returns:
            float: cardinal degree
        """
        angle = self.slope_angle(degree=True, is_cv2=True)
        
        # if angle is negative
        is_neg_sign_angle = bool(np.sign(angle)-1)
        if is_neg_sign_angle:
            angle = 90 + np.abs(angle)
        else:
            angle = 90 - angle
        
        # if vector points towards west
        if self.is_x0_greater_than_x1:
            angle += 180
            
        cardinal_degree = np.mod(360 + angle, 360) # avoid negative value case
        return cardinal_degree
    
    def cardinal_direction(self, full: bool=False, level: int=2):
        return angle_to_direction(input_angle=self.cardinal_degree, full=full, level=level)