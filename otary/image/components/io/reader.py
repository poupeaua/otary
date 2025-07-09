"""
Image Reader module
"""

from typing import Optional

import cv2
import numpy as np
from numpy.typing import NDArray
import pymupdf

from otary.image.components.io.utils.readfile import read_pdf_to_images


class ReaderImage:
    """ReaderImage class to facilitate the reading of images from different formats
    such as JPG, PNG, and PDF. It provides methods to load images from file paths.
    """

    @staticmethod
    def from_fillvalue(value: int = 255, shape: tuple = (128, 128, 3)) -> NDArray:
        """Create an array image from a single value

        Args:
            value (int, optional): value in [0, 255]. Defaults to 255.
            shape (tuple, optional): image shape. If it has three elements then
                the last one must be a 3 for a coloscale image.
                Defaults to (128, 128, 3).

        Returns:
            NDArray: array with a single value
        """
        if value < 0 or value > 255:
            raise ValueError(f"The value {value} must be in [0, 255]")
        if len(shape) < 2 or len(shape) >= 4:
            raise ValueError(f"The shape {shape} must be of length 2 or 3")
        if len(shape) == 3 and shape[-1] != 3:
            raise ValueError(f"The last value of {shape} must be 3")
        return np.full(shape=shape, fill_value=value, dtype=np.uint8)

    @staticmethod
    def from_jpg(
        filepath: str, as_grayscale: bool = False, resolution: Optional[int] = None
    ) -> NDArray:
        """Create a Image object from a JPG or JPEG file path

        Args:
            filepath (str): path to the JPG image file
            as_grayscale (bool, optional): turn the image in grayscale.
                Defaults to False.

        Returns:
            Self: Image object
        """
        arr = np.asarray(cv2.imread(filepath, 1 - int(as_grayscale)))
        original_height, original_width = arr.shape[:2]

        if resolution is not None:
            # Calculate the aspect ratio
            aspect_ratio = original_width / original_height
            new_width = int(resolution * aspect_ratio)
            arr = cv2.resize(src=arr, dsize=(new_width, resolution))

        return arr

    @staticmethod
    def from_png(
        filepath: str, as_grayscale: bool = False, resolution: Optional[int] = None
    ) -> NDArray:
        """Create a Image array from a PNG file image path

        Args:
            filepath (str): path to the image file
            as_grayscale (bool, optional): turn the image in grayscale.
                Defaults to False.

        Returns:
            NDArray: Image as array
        """
        return ReaderImage.from_jpg(
            filepath=filepath, as_grayscale=as_grayscale, resolution=resolution
        )

    @staticmethod
    def from_pdf(
        filepath: str,
        as_grayscale: bool = False,
        page_nb: int = 0,
        resolution: Optional[int] = None,
        clip_pct: Optional[pymupdf.Rect] = None,
    ) -> NDArray:
        # pylint: disable=too-many-arguments, too-many-positional-arguments
        """Create an Image array from a pdf file.

        Args:
            filepath (str): path to the pdf file.
            as_grayscale (bool, optional): whether to turn the image in grayscale.
                Defaults to False.
            page_nb (int, optional): as we load only one image we have to select the
                page that will be turned into an image. Defaults to 0.
            resolution (Optional[int], optional): resolution of the loaded image.
                Defaults to 3508.
            clip_pct (pymmupdf.Rect, optional): optional zone to extract in the image.
                This is particularly useful to load into memory only a small part of the
                image without loading everything into memory. This reduces considerably
                the image loading time especially combined with a high resolution.

        Returns:
            NDArray: Image as array
        """
        arr = read_pdf_to_images(
            filepath_or_stream=filepath,
            resolution=resolution,
            page_nb=page_nb,
            clip_pct=clip_pct,
        )[0]

        if as_grayscale:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)

        return arr

    @staticmethod
    def from_file(
        filepath: str, as_grayscale: bool = False, resolution: Optional[int] = None
    ) -> NDArray:
        """Create a Image array from a file image path

        Args:
            filepath (str): path to the image file
            as_grayscale (bool, optional): turn the image in grayscale.
                Defaults to False.

        Returns:
            NDArray: Image as array
        """
        valid_format = ["png", "jpg", "jpeg", "pdf"]

        file_format = filepath.split(".")[-1]

        if file_format in ["png"]:
            return ReaderImage.from_png(
                filepath=filepath, as_grayscale=as_grayscale, resolution=resolution
            )
        if file_format in ["jpg", "jpeg"]:
            return ReaderImage.from_jpg(
                filepath=filepath, as_grayscale=as_grayscale, resolution=resolution
            )
        if file_format in ["pdf"]:
            return ReaderImage.from_pdf(
                filepath=filepath, as_grayscale=as_grayscale, resolution=resolution
            )

        raise ValueError(f"The filepath is not in any valid format {valid_format}")
