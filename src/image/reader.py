"""
Image Reader module
"""

from abc import ABC
from typing import Optional, Self

import cv2
import numpy as np
import pymupdf

from src.image.utils.readfile import read_pdf_to_images
from src.image.base import BaseImage


class ReaderImage(BaseImage, ABC):
    """ReaderImage class to facilitate the reading of images from different formats
    such as JPG, PNG, and PDF. It provides methods to load images from file paths.
    """

    @classmethod
    def from_jpg(
        cls, filepath: str, as_grayscale: bool = False, resolution: Optional[int] = None
    ) -> Self:
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

        return cls(arr)

    @classmethod
    def from_png(
        cls, filepath: str, as_grayscale: bool = False, resolution: Optional[int] = None
    ) -> Self:
        """Create a Image object from a PNG file image path

        Args:
            filepath (str): path to the image file
            as_grayscale (bool, optional): turn the image in grayscale.
                Defaults to False.

        Returns:
            Self: Image object
        """
        return cls.from_jpg(
            filepath=filepath, as_grayscale=as_grayscale, resolution=resolution
        )

    @classmethod
    def from_pdf(
        cls,
        filepath: str,
        as_grayscale: bool = False,
        page_nb: int = 0,
        resolution: Optional[int] = None,
        clip_pct: Optional[pymupdf.Rect] = None,
    ) -> Self:
        # pylint: disable=too-many-arguments, too-many-positional-arguments
        """Create an Image object from a pdf file.

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
            Self: Image object
        """
        arr = read_pdf_to_images(
            filepath_or_stream=filepath,
            resolution=resolution,
            page_nb=page_nb,
            clip_pct=clip_pct,
        )[0]

        if as_grayscale:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)

        return cls(arr)

    @classmethod
    def from_file(
        cls, filepath: str, as_grayscale: bool = False, resolution: Optional[int] = None
    ) -> Self:
        """Create a Image object from a file image path

        Args:
            filepath (str): path to the image file
            as_grayscale (bool, optional): turn the image in grayscale.
                Defaults to False.

        Returns:
            Self: Image object
        """
        valid_format = ["png", "jpg", "jpeg", "pdf"]

        file_format = filepath.split(".")[-1]

        if file_format in ["png"]:
            return cls.from_png(
                filepath=filepath, as_grayscale=as_grayscale, resolution=resolution
            )
        if file_format in ["jpg", "jpeg"]:
            return cls.from_jpg(
                filepath=filepath, as_grayscale=as_grayscale, resolution=resolution
            )
        if file_format in ["pdf"]:
            return cls.from_pdf(
                filepath=filepath, as_grayscale=as_grayscale, resolution=resolution
            )

        raise ValueError(f"The filepath is not in any valid format {valid_format}")
