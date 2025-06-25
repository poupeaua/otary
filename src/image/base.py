"""
Base Image module for basic image processing.
It only contains very low-level, basic and generic image methods.
"""

from __future__ import annotations

import io
from typing import Self

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image as ImagePIL


class BaseImage:
    """Base Image class"""

    # pylint: disable=too-many-public-methods

    def __init__(self, image: NDArray) -> None:
        self.__asarray: NDArray = image.copy()

    @property
    def asarray(self) -> NDArray:
        """Array representation of the image"""
        return self.__asarray

    @asarray.setter
    def asarray(self, value: NDArray) -> None:
        """Setter for the asarray property

        Args:
            value (np.ndarray): value of the asarray to be changed
        """
        self.__asarray = value

    @property
    def asarray_binary(self) -> NDArray:
        """Returns the representation of the image as a array with value not in
        [0, 255] but in [0, 1].

        Returns:
            NDArray: an array with value in [0, 1]
        """
        return (self.asarray / 255).astype(np.float32)

    @property
    def is_gray(self) -> bool:
        """Whether the image is a grayscale image or not

        Returns:
            bool: True if image is in grayscale, 0 otherwise
        """
        return bool(len(self.asarray.shape) == 2)

    @property
    def channels(self) -> int:
        """Number of channels in the image

        Returns:
            int: number of channels
        """
        if self.is_gray:
            return 1
        return self.asarray.shape[2]

    @property
    def shape_array(self) -> tuple:
        """Returns the array shape value (height, width, channel)

        Returns:
            tuple[int]: image shape
        """
        return self.asarray.shape

    @property
    def shape(self) -> tuple:
        """Returns the array shape value (width, height, channel)

        Returns:
            tuple[int]: image shape
        """
        return (self.width, self.height, self.channels)

    @property
    def height(self) -> int:
        """Height of the image.

        Returns:
            int: image height
        """
        return self.asarray.shape[0]

    @property
    def width(self) -> int:
        """Width of the image.

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
    def center(self) -> NDArray[np.int16]:
        """Center point of the image.

        Please note that it is returned as type int because the center is
        represented as a X-Y coords of a pixel.

        Returns:
            np.ndarray: center point of the image
        """
        return (np.array([self.width, self.height]) / 2).astype(np.int16)

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
    def corners(self) -> NDArray:
        """Returns the corners in clockwise order:

        0. top left corner
        1. top right corner
        2. bottom right corner
        3. bottom left corner

        Returns:
            NDArray: array containing the corners
        """
        return np.array(
            [self.top_left, self.top_right, self.bottom_right, self.bottom_left]
        )

    @property
    def bottom_right(self) -> NDArray:
        """Get the bottom right point coordinate of the image

        Returns:
            NDArray: 2D point
        """
        return np.array([self.width - 1, self.height - 1], dtype=int)

    @property
    def bottom_left(self) -> NDArray:
        """Get the bottom right point coordinate of the image

        Returns:
            NDArray: 2D point
        """
        return np.array([0, self.height - 1], dtype=int)

    @property
    def top_right(self) -> NDArray:
        """Get the bottom right point coordinate of the image

        Returns:
            NDArray: 2D point
        """
        return np.array([self.width - 1, 0], dtype=int)

    @property
    def top_left(self) -> NDArray:
        """Get the bottom right point coordinate of the image

        Returns:
            NDArray: 2D point
        """
        return np.array([0, 0], dtype=int)

    def as_pil(self) -> ImagePIL.Image:
        """Return the image as PIL Image

        Returns:
            ImagePIL: PIL Image
        """
        return ImagePIL.fromarray(self.asarray)

    def as_bytes(self, fmt: str = "PNG") -> bytes:
        """Return the image as bytes

        Args:
            fmt (str, optional): format of the image. Defaults to "PNG".

        Returns:
            bytes: image in bytes
        """
        pil_image = self.as_pil()
        with io.BytesIO() as output:
            pil_image.save(output, format=fmt)
            return output.getvalue()

    def as_api_file_input(
        self, fmt: str = "png", filename: str = "image"
    ) -> dict[str, tuple[str, bytes, str]]:
        """Return the image as a file input for API requests.

        Args:
            fmt (str, optional): format of the image. Defaults to "png".
            filename (str, optional): name of the file. Defaults to "image".

        Returns:
            dict[str, tuple[str, bytes, str]]: dictionary with file input
                for API requests, where the key is "file" and the value is a tuple
                containing the filename, image bytes, and content type.
        """
        fmt_lower = fmt.lower()
        files = {
            "file": (
                f"{filename}.{fmt_lower}",
                self.as_bytes(fmt=fmt),
                f"image/{fmt_lower}",
            )
        }
        return files

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

    def as_black(self) -> Self:
        """Returns an entirely black image with the same dimension as the original.

        Returns:
            Self: new black image
        """
        self.as_filled(fill_value=0)
        return self

    def rev(self) -> Self:
        """Reverse the image colors. Each pixel color value V becomes |V - 255|.

        Applied on a grayscale image the black pixel becomes white and the
        white pixels become black.
        """
        self.asarray = np.abs(self.asarray.astype(np.int16) - 255).astype(np.uint8)
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

    def dist_pct(self, pct: float) -> float:
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
