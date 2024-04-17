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