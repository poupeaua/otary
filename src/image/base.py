"""
Base Image module for basic image processing.
It only contains very low-level, basic and generic image methods.
"""

from __future__ import annotations

from typing import Self
from abc import ABC

import cv2
import numpy as np
from PIL import Image as ImagePIL


class BaseImage(ABC):
    """Base Image class"""

    # pylint: disable=too-many-public-methods

    def __init__(self, image: np.ndarray | BaseImage) -> None:
        if isinstance(image, BaseImage):
            image = image.asarray
        self.__asarray: np.ndarray = image.copy()

    @classmethod
    def from_fillvalue(cls, value: int = 255, shape: tuple = (128, 128, 3)) -> Self:
        """Class method to create an image from a single value

        Args:
            value (int, optional): value in [0, 255]. Defaults to 255.
            shape (tuple, optional): image shape. If it has three elements then
                the last one must be a 3 for a coloscale image.
                Defaults to (128, 128, 3).

        Returns:
            Self: new Image object with a single value
        """
        if value < 0 or value > 255:
            raise ValueError(f"The value {value} must be in [0, 255]")
        if len(shape) < 2 or len(shape) >= 4:
            raise ValueError(f"The shape {shape} must be of length 2 or 3")
        if len(shape) == 3 and shape[-1] != 3:
            raise ValueError(f"The last value of {shape} must be 3")
        return cls(np.full(shape=shape, fill_value=value, dtype=np.uint8))

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
    def shape_array(self) -> tuple:
        """Returns the array shape value (height, width, channel)

        Returns:
            tuple[int]: image shape
        """
        return self.asarray.shape

    @property
    def height(self) -> int:
        """Height of the image. In numpy it is defined as the first image shape value

        Returns:
            int: image height
        """
        return self.asarray.shape[0]

    @property
    def width(self) -> int:
        """Width of the image. In numpy it is defined as the second image shape value

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
        return (self.asarray / 255).astype(np.float32)

    @property
    def corners(self) -> np.ndarray:
        """Returns the corners in the following order:

        0. top left corner
        1. top right corner
        2. bottom right corner
        3. bottom left corner

        Returns:
            np.ndarray: array containing the corners
        """
        return np.array(
            [self.top_left, self.top_right, self.bottom_right, self.bottom_left]
        )

    @property
    def bottom_right(self) -> np.ndarray:
        """Get the bottom right point coordinate of the image

        Returns:
            np.ndarray: 2D point
        """
        return np.array([self.width, self.height], dtype=int)

    @property
    def bottom_left(self) -> np.ndarray:
        """Get the bottom right point coordinate of the image

        Returns:
            np.ndarray: 2D point
        """
        return np.array([0, self.height], dtype=int)

    @property
    def top_right(self) -> np.ndarray:
        """Get the bottom right point coordinate of the image

        Returns:
            np.ndarray: 2D point
        """
        return np.array([self.width, 0], dtype=int)

    @property
    def top_left(self) -> np.ndarray:
        """Get the bottom right point coordinate of the image

        Returns:
            np.ndarray: 2D point
        """
        return np.array([0, 0], dtype=int)

    def as_pil(self) -> ImagePIL.Image:
        """Return the image as PIL Image

        Returns:
            ImagePIL: PIL Image
        """
        return ImagePIL.fromarray(self.asarray)

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
        self.asarray = np.full(
            shape=self.shape_array, fill_value=fill_value, dtype=np.uint8
        )
        return self

    def as_white(self) -> Self:
        """Returns an entirely white image with the same dimension as the original.

        Returns:
            Self: new white image
        """
        self.as_filled(fill_value=255)
        return self

    def is_equal_shape(self, other: BaseImage, consider_channel: bool = True) -> bool:
        """Check whether two images have the same shape

        Args:
            other (BaseImage): BaseImage object

        Returns:
            bool: True if the objects have the same shape, False otherwise
        """
        if consider_channel:
            shape0 = self.shape_array
            shape1 = other.shape_array
        else:
            shape0 = (
                self.shape_array
                if len(self.shape_array) == 2
                else self.shape_array[:-1]
            )
            shape1 = (
                self.shape_array
                if len(self.shape_array) == 2
                else self.shape_array[:-1]
            )
        return shape0 == shape1

    def dist_pct(self, pct: float = 0.01) -> float:
        """Distance percentage that can be used an acceptable distance error margin.
        It is calculated based on the normalized side length.

        Args:
            pct (float, optional): percentage of distance error. Defaults to 0.01,
                which means 1% of the normalized side length as the
                default margin distance error.

        Returns:
            float: margin distance error
        """
        return self.norm_side_length * pct

    def width_pct(self, pct: float = 0.01) -> float:
        """Width percentage of the image

        Args:
            pct (float, optional): percentage of width. Defaults to 0.01.

        Returns:
            float: Width percentage of the image value distance
        """
        return self.width * pct

    def height_pct(self, pct: float = 0.01) -> float:
        """Height percentage of the image

        Args:
            pct (float, optional): percentage of height. Defaults to 0.01.

        Returns:
            float: Height percentage of the image value distance
        """
        return self.height * pct

    def copy(self) -> Self:
        """Copy of the image.

        For NumPy arrays containing basic data types (e.g., int, float, bool),
        using copy.deepcopy() is generally unnecessary.
        The numpy.copy() method achieves the same result more efficiently.
        numpy.copy() creates a new array in memory with a separate copy of the data,
        ensuring that modifications to the copy do not affect the original array.

        Returns:
            Image: image copy
        """
        return type(self)(image=self.asarray.copy())
