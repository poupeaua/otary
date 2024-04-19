"""
Segment class to describe defined lines and segments
"""

import numpy as np
import cv2
from src.geometry.entity import GeometryEntity

class Segment(GeometryEntity):
    
    def __init__(self, points) -> None:
        super().__init__(points)
        
    @property
    def perimeter(self):
        return cv2.arcLength(self.points, False)
        
    @property
    def length(self):
        return self.perimeter
    
    @property
    def centroid(self) -> float:
        return np.sum(self.points, axis=0) / 2
    
    @property
    def slope(self) -> float:
        p1, p2 = self.points[0], self.points[1]
        try:
            slope = (p2[1] - p1[1]) / (p2[0] - p1[0] + 1e-9)
        except ZeroDivisionError:
            slope = np.inf
        return slope
    
    @property
    def slope_cv2(self) -> float:
        return -self.slope 
    
    def slope_angle(self, degree: bool=False, is_cv2: bool=False) -> float:
        """Calculate the slope angle of a single line in the cartesian space

        Args:
            degree (bool): whether to output the result in degree. By default in radian.

        Returns:
            float: slope angle in ]-pi/2, pi/2[
        """
        angle = np.arctan(self.slope_cv2) if is_cv2 else np.arctan(self.slope)
        if degree:
            angle = np.rad2deg(angle)
        return angle