"""
ContinuousGeometryEntity module class
"""

from __future__ import annotations

from typing import Optional, Self
import copy
from abc import ABC, abstractmethod

import cv2
import numpy as np

from src.geometry.entity import GeometryEntity
from src.geometry import Polygon


class ContinuousGeometryEntity(GeometryEntity, ABC):
    """
    ContinuousGeometryEntity class which is the abstract base class for
    continuous or smooth geometry objects like circles, ellipse, etc...
    """

    # --------------------------------- PROPERTIES ------------------------------------

    @abstractmethod
    def polygonal_approx(self, n_points: int) -> Polygon:
        """Generate a polygonal approximation of the continuous geometry entity

        Args:
            n_points (int): number of points that make up the polygonal
                approximation. The bigger the better to obtain more precise
                results in intersection or other similar computations.

        Returns:
            Polygon: polygonal approximation of the continuous geometry entity
        """

    @property
    def area(self) -> float:
        """Compute the area of the geometry entity

        Returns:
            float: area value
        """
        return cv2.contourArea(self.points.astype(int))

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
        return np.mean(self.points, axis=0)

    @property
    def xmax(self) -> float:
        """Get the maximum X coordinate of the geometry entity

        Returns:
            np.ndarray: 2D point
        """
        # TODO

    @property
    def xmin(self) -> float:
        """Get the minimum X coordinate of the geometry entity

        Returns:
            np.ndarray: 2D point
        """
        # TODO

    @property
    def ymax(self) -> float:
        """Get the maximum Y coordinate of the geometry entity

        Returns:
            np.ndarray: 2D point
        """
        # TODO

    @property
    def ymin(self) -> float:
        """Get the minimum Y coordinate of the geometry entity

        Returns:
            np.ndarray: 2D point
        """
        # TODO

    # ---------------------------- MODIFICATION METHODS -------------------------------

    def rotate(
        self,
        angle: float,
        is_degree: bool = False,
        is_clockwise: bool = True,
        pivot: Optional[np.ndarray] = None,
    ) -> Self:
        """Rotate the geometry entity object.
        A pivot point can be passed as an argument to rotate the object around the pivot

        Args:
            angle (float): rotation angle
            is_degree (bool, optional): whether the angle is in degree or radian.
                Defaults to False which means radians.
            is_clockwise (bool, optional): whether the rotation is clockwise or
                counter-clockwise. Defaults to True.
            pivot (np.ndarray, optional): pivot point.
                Defaults to None which means that by default the centroid point of
                the shape is taken as the pivot point.

        Returns:
            GeometryEntity: rotated geometry entity object.
        """
        if pivot is None:
            pivot = self.centroid

        if is_degree:  # transform angle to radian if in degree
            angle = np.deg2rad(angle)

        if not is_clockwise:
            angle = -angle

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
    ) -> Self:
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
        return self.rotate(angle=angle, pivot=img_center_point, is_degree=degree)

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

    def shift(self, vector: np.ndarray) -> Self:
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

    def clamp(
        self,
        xmin: float = -np.inf,
        xmax: float = np.inf,
        ymin: float = -np.inf,
        ymax: float = np.inf,
    ) -> Self:
        """Clamp the Geometry entity so that the x and y coordinates fit in the
        min and max values in parameters.

        Args:
            xmin (float): x coordinate minimum
            xmax (float): x coordinate maximum
            ymin (float): y coordinate minimum
            ymax (float): y coordinate maximum

        Returns:
            GeometryEntity: clamped GeometryEntity
        """
        self.asarray[:, 0] = np.clip(self.asarray[:, 0], xmin, xmax)  # x values
        self.asarray[:, 1] = np.clip(self.asarray[:, 1], ymin, ymax)  # y values
        return self

    def normalize(self, x: float, y: float) -> Self:
        """Normalize the geometry entity by dividing the points by a norm on the
        x and y coordinates.

        Args:
            x (float): x coordinate norm
            y (float): y coordinate norm

        Returns:
            GeometryEntity: normalized GeometryEntity
        """
        self.asarray = self.asarray / np.array([x, y])
        return self

    # ------------------------------- CLASSIC METHODS ---------------------------------

    def copy(self) -> Self:
        """Create a copy of the geometry entity object

        Returns:
            GeometryEntity: copy of the geometry entity object
        """
        return type(self)(
            points=copy.deepcopy(self.asarray), is_cast_int=self.is_cast_int
        )

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, ContinuousGeometryEntity):
            return False
        return np.array_equal(self.asarray, value.asarray)

    def __add__(self, other: np.ndarray | float) -> Self:
        self.asarray += other
        return self

    def __sub__(self, other: np.ndarray | float) -> Self:
        self.asarray -= other
        return self

    def __mul__(self, other: np.ndarray | float) -> Self:
        self.asarray *= other
        return self

    def __truediv__(self, other: np.ndarray | float) -> Self:
        self.asarray = self.asarray / other
        return self

    def __len__(self) -> int:
        return self.n_points

    def __str__(self) -> str:
        return self.__class__.__name__ + "(" + self.asarray.tolist().__str__() + ")"

    def __repr__(self) -> str:
        return self.__class__.__name__ + "(" + self.asarray.tolist().__repr__() + ")"
