"""
Base Image module for basic image processing.
It only contains very low-level, basic and generic image methods.
"""

from __future__ import annotations

from typing import Self
from abc import ABC

import cv2
import numpy as np


class BaseImage(ABC):
    """Base Image class"""

    def __init__(self, image: np.ndarray | BaseImage) -> None:
        if isinstance(image, BaseImage):
            image = image.asarray
        self.__asarray: np.ndarray = image.copy()

    @property
    def asarray(self) -> np.ndarray:
        """Array representation of the image"""
        return self.__asarray

    @asarray.setter
    def asarray(self, value: np.ndarray):
        """Setter for the asarray property

        Args:
            value (np.ndarray): value of the asarray to be changed
        """
        self.__asarray = value

    @property
    def is_gray(self) -> bool:
        """Whether the image is a grayscale image or not

        Returns:
            bool: True if image is in grayscale, 0 otherwise
        """
        return bool(len(self.asarray.shape) == 2)

    @property
    def shape(self) -> tuple:
        """Returns the image shape value

        Returns:
            tuple[int]: image shape
        """
        return self.asarray.shape

    @property
    def height(self) -> int:
        """Height of the image. In cv2 it is defined as the first image shape value

        Returns:
            int: image height
        """
        return self.asarray.shape[0]

    @property
    def width(self) -> int:
        """Width of the image. In cv2 it is defined as the second image shape value

        Returns:
            int: image width
        """
        return self.asarray.shape[1]

    @property
    def area(self) -> int:
        """Area of the image

        Returns:
            int: image area
        """
        return self.width * self.height

    @property
    def center(self) -> np.ndarray:
        """Center point of the image.

        Please note that it is returned as type int because the center needs to
        represent a X-Y coords of a pixel.

        Returns:
            np.ndarray: center point of the image
        """
        return (np.array([self.width, self.height]) / 2).astype(int)

    @property
    def norm_side_length(self) -> int:
        """Returns the normalized side length of the image.
        This is the side length if the image had the same area but
        the shape of a square (four sides of the same length).

        Returns:
            int: normalized side length
        """
        return int(np.sqrt(self.area))

    @property
    def asarray_norm(self) -> np.ndarray:
        """Returns the representation of the image as a array with value not in
        [0, 255] but in [0, 1].

        Returns:
            np.ndarray: an array with value in [0, 1]
        """
        return (self.asarray / 255).astype(np.uint8)

    @property
    def corners(self) -> np.ndarray:
        """Returns the corners in the following order:

        0. bottom left corner
        1. bottom right corner
        2. top right corner
        3. top left corner

        Returns:
            np.ndarray: array containing the corners
        """
        return np.array(
            [[0, 0], [self.width, 0], [self.width, self.height], [0, self.height]]
        )

    def as_grayscale(self) -> Self:
        """Generate the image in grayscale of shape (height, width)

        Returns:
            Self: original image in grayscale
        """
        if self.is_gray:
            return self
        self.asarray = cv2.cvtColor(self.asarray, cv2.COLOR_BGR2GRAY)
        return self

    def as_colorscale(self) -> Self:
        """Generate the image in colorscale (height, width, 3).
        This property can be useful when we wish to draw objects in a given color
        on a grayscale image.

        Returns:
            Self: original image in color
        """
        if not self.is_gray:
            return self
        self.asarray = cv2.cvtColor(self.asarray, cv2.COLOR_GRAY2BGR)
        return self

    def as_filled(self, fill_value: int | np.ndarray = 255) -> Self:
        """Returns an entirely white image of the same size as the original.
        Can be useful to get an empty representation of the same image to paint
        and draw things on an image of the same dimension.

        Args:
            fill_value (int | np.ndarray, optional): color to fill the new empty image.
                Defaults to 255 which means that is returns a entirely white image.

        Returns:
            Self: new image with a single color of the same size as original.
        """
        self.asarray = np.full(shape=self.shape, fill_value=fill_value, dtype=np.uint8)
        return self

    def as_white(self) -> Self:
        """Returns an entirely white image with the same dimension as the original.

        Returns:
            Self: new white image
        """
        self.as_filled(fill_value=255)
        return self

    def is_equal_shape(self, other: BaseImage) -> bool:
        """Check whether two images have the same shape

        Args:
            other (BaseImage): BaseImage object

        Returns:
            bool: True if the objects have the same shape, False otherwise
        """
        return self.shape == other.shape

    def dist_pct(self, pct: float = 0.01) -> float:
        """Distance percentage that can be used an acceptable distance error margin.
        It is calculated based on the normalized side length.

        Args:
            pct (float, optional): pourcentage of distance error. Defaults to 0.01,
                which means 1% of the normalized side length as the
                default margin distance error.

        Returns:
            float: margin distance error
        """
        return self.norm_side_length * pct
