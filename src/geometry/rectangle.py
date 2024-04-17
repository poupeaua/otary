"""
Rectangle class.
It will be particularly useful for the AITT project for describing bounding boxes.
"""

import numpy as np
from src.geometry.contour import Contour


class Rectangle(Contour):
    
    def __init__(self, points: np.ndarray) -> None:
        super().__init__(points, reduce=False)
    
    @classmethod
    def from_center(cls, center: np.ndarray, width: float, height: float):
        # compute the halves lengths
        half_width = width / 2
        half_height = height / 2
        
        # get center coordinates
        center_x, center_y = center[0], center[1]
        
        # get the rectangle coordinates
        top_left_corner = np.array([center_x - half_width, center_y + half_height])
        top_right_corner = np.array([center_x + half_width, center_y + half_height])
        bottom_left_corner = np.array([center_x - half_width, center_y - half_height])
        bottom_right_corner = np.array([center_x + half_width, center_y - half_height])
        rectangle = np.array([top_left_corner, top_right_corner, 
                            bottom_left_corner, bottom_right_corner])
        
        return rectangle