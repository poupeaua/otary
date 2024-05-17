"""

"""

from abc import ABC
import cv2
import numpy as np
import copy


class GeometryEntity(ABC):
    
    def __init__(self, points) -> None:
        self.points = copy.deepcopy(np.array(points))
          
    @property
    def asarray(self) -> np.ndarray:
        return self.points
    
    @property
    def area(self) -> float:
        return cv2.contourArea(self.points)
    
    @property
    def perimeter(self) -> float:
        return cv2.arcLength(self.points, True)
    
    @property
    def centroid(self) -> float:
        moments = cv2.moments(self.points)
        if moments['m00'] != 0:
            centroid = np.array([moments['m10']/moments['m00'], moments['m01']/moments['m00']])
        else:
            centroid = None
        return centroid
    
    def copy(self):
        return copy.deepcopy(self)
    
    def rotate(
            self,
            angle: float, 
            pivot: np.ndarray=np.zeros(shape=(2,)),
            degree: bool=False
        ):
        if degree: # transform angle to radian if in degree
            angle = np.deg2rad(angle)
        
        # Translate the point so that the pivot is at the origin
        translated_points = self.points - pivot
        
        # Define the rotation matrix
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle), np.cos(angle)]])
        
        # Apply the rotation matrix to translated point
        rotated_points = np.matmul(rotation_matrix, translated_points.T).T
        
        # Translate the point back to its original space and return
        self.points = rotated_points + pivot
        return self
        
    def rotate_around_image_center(
            self,
            img: np.ndarray,
            angle: float, 
            degree: bool=False
        ):
        img_center_point = np.array([img.shape[1], img.shape[0]]) / 2
        return self.rotate(angle=angle, pivot=img_center_point, degree=degree)
        
    
    def shift(
            self,
            vector: np.ndarray
        ):
        self.points = self.points + vector
        return self
    
    def __str__(self) -> str:
        return self.__class__.__name__ + "(" + self.asarray.tolist().__str__() + ")"
    
    def __repr__(self) -> str:
        return self.__class__.__name__ + "(" + self.asarray.tolist().__repr__() + ")"