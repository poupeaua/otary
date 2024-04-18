"""
Point class useful to describe any kind of points
"""

import numpy as np
from src.geometry.entity import GeometryEntity


class Point(GeometryEntity):
    
    def __init__(self, point: np.ndarray) -> None:
        super().__init__(points=point)
    
    @property
    def asarray(self):
        return self.points[0]