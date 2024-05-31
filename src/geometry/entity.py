"""
GeometryEntity module which allows to define transformation and property shared
by all type of geometry objects
"""

from __future__ import annotations

from typing import Optional
import copy
from abc import ABC

import cv2
import numpy as np


class GeometryEntity(ABC):
    """GeometryEntity class which is the abstract base class for all geometry classes"""

    def __init__(self, points) -> None:
        self.points = copy.deepcopy(np.array(points))

    @property
    def n_points(self) -> int:
        """Returns the number of points this geometric object is made of

        Returns:
            int: number of points that composes the geomtric object
        """
        return self.points.shape[0]

    @property
    def asarray(self) -> np.ndarray:
        """Representation of the object as a numpy array

        Returns:
            np.ndarray: numpy array representation of the object
        """
        return self.points

    @property
    def area(self) -> float:
        """Compute the area of the geometry entity

        Returns:
            float: area value
        """
        return cv2.contourArea(self.points)

    @property
    def perimeter(self) -> float:
        """Compute the perimeter of the geometry entity

        Returns:
            float: perimeter value
        """
        return cv2.arcLength(self.points, True)

    @property
    def centroid(self) -> np.ndarray:
        """Compute the centroid point which can be seen as the center of gravity of
        the shape

        Returns:
            np.ndarray: centroid point
        """
        moments = cv2.moments(self.points)
        if moments["m00"] != 0:
            centroid = np.array(
                [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]]
            )
        else:
            centroid = np.mean(self.points, axis=0)  # useful for the point entity
        return centroid

    def copy(self):
        """Create a copy of the geometry entity object

        Returns:
            GeometryEntity: copy of the geometry entity object
        """
        return copy.deepcopy(self)

    def rotate(
        self, angle: float, degree: bool = False, pivot: Optional[np.ndarray] = None
    ):
        """Rotate the geometry entity object.
        A pivot point can be passed as an argument to rotate the object around the pivot

        Args:
            angle (float): rotation angle
            pivot (np.ndarray, optional): pivot point.
                Defaults to None which means that by default the centroid point of
                the shape is taken as the pivot point.
            degree (bool, optional): whether the angle is in degree or radian.
                Defaults to False which means radians.

        Returns:
            GeometryEntity: rotated geometry entity object.
        """
        if pivot is None:
            pivot = self.centroid

        if degree:  # transform angle to radian if in degree
            angle = np.deg2rad(angle)

        # Translate the point so that the pivot is at the origin
        translated_points = self.points - pivot

        # Define the rotation matrix
        rotation_matrix = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )

        # Apply the rotation matrix to translated point
        rotated_points = np.matmul(rotation_matrix, translated_points.T).T

        # Translate the point back to its original space and return
        self.points = rotated_points + pivot
        return self

    def rotate_around_image_center(
        self, img: np.ndarray, angle: float, degree: bool = False
    ):
        """Given an geometric object and an image, rotate the object around
        the image center point.

        Args:
            img (np.ndarray): image as a shape (x, y) sized array
            angle (float): rotation angle
            degree (bool, optional): whether the angle is in degree or radian.
                Defaults to False which means radians.

        Returns:
            GeometryEntity: rotated geometry entity object.
        """
        img_center_point = np.array([img.shape[1], img.shape[0]]) / 2
        return self.rotate(angle=angle, pivot=img_center_point, degree=degree)

    def __validate_shift_vector(self, vector: np.ndarray) -> np.ndarray:
        """Validate the shift vector before executing operation

        Args:
            vector (np.ndarray): shift vector

        Returns:
            np.ndarray: validated vector
        """
        vector = np.asarray(vector)
        if vector.shape == (2, 2):
            vector = vector[1] - vector[0]  # set the vector to be defined as one point
        if vector.shape != (2,):
            raise ValueError(
                "The input vector {vector} does not have the expected shape."
            )
        return vector

    def shift(self, vector: np.ndarray):
        """Shift the geometry entity by the vector direction

        Args:
            vector (np.ndarray): vector that describes the shift as a array with
                two elements. Example: [2, -8] which describes the
                vector [[0, 0], [2, -8]]. The vector can also be a vector of shape
                (2, 2) of the form [[2, 6], [1, 3]].

        Returns:
            GeometryEntity: shifted geometrical object
        """
        vector = self.__validate_shift_vector(vector=vector)
        self.points = self.points + vector
        return self

    def __str__(self) -> str:
        return self.__class__.__name__ + "(" + self.asarray.tolist().__str__() + ")"

    def __repr__(self) -> str:
        return self.__class__.__name__ + "(" + self.asarray.tolist().__repr__() + ")"
