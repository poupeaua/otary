"""
Image Trasnformation module. it only contains advanced image transformation methods.
"""

from __future__ import annotations

from typing import Self

from abc import ABC
import cv2
import numpy as np
import scipy.ndimage

# pylint: disable=no-name-in-module
from skimage.filters import threshold_sauvola, threshold_otsu

import src.geometry as geo
from src.image.base import BaseImage


class TransformerImage(BaseImage, ABC):
    """Transformer images utility class"""

    def binary(self, method: str = "sauvola") -> np.ndarray:
        """Binary representation of the image with values that can be only 0 or 1.
        The value 0 is now 0 and value of 255 are now 1. Black is 0 and white is 1.

        Args:
            method (str, optional): the binarization method to apply.
                Defaults to "sauvola".

        Returns:
            np.ndarray: array where its inner values are 0 or 1
        """
        valid_binarization_methods = ["otsu", "sauvola"]

        if method == "otsu":
            return self.threshold_otsu().asarray_norm
        if method == "sauvola":
            return self.threshold_sauvola().asarray_norm

        raise ValueError(f"The method {method} is not in {valid_binarization_methods}")

    def binaryrev(self, method: str = "sauvola") -> np.ndarray:
        """Reversed binary representation of the image.
        The value 0 is now 1 and value of 255 are now 0. Black is 1 and white is 0.
        This is why it is called the "binary rev" or "binary reversed".

        Args:
            method (str, optional): the binarization method to apply.
                Defaults to "sauvola".

        Returns:
            np.ndarray: array where its inner values are 0 or 1
        """
        return 1 - self.binary(method=method)

    def rev(self) -> Self:
        """Reverse the image

        Returns:
            Self: the Image in reversed. The black pixel becomes white and the
                white pixels become black.
        """
        self.as_grayscale()
        self.asarray = np.asarray(self.binaryrev() * 255, dtype=np.uint8)
        return self

    def threshold_simple(self, threshold_value: int) -> Self:
        """Compute the image thesholded by a single value T.
        All pixels with value v < T are turned black and those with value v > T are
        turned white.

        Args:
            threshold_value (int): value to separate the black from the white pixels.

        Returns:
            Self: new image thresholded
        """
        self.as_grayscale()
        self.asarray = np.asarray((self.asarray > threshold_value) * 255).astype(
            np.uint8
        )
        return self

    def threshold_otsu(
        self, is_blur_enabled: bool = False, blur_ksize: int = 5
    ) -> Self:
        """Apply Ostu thresholding.
        A blur is applied before for better thresholding results.
        See https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html.

        As the input image must be a grayscale before applying any thresholding
        methods we convert the image to grayscale.

        Args:
            is_blur_enabled (bool, optional): whether to apply blur before applying
                the otsu thresholding. Defaults to False.
            blur_ksize (int, optional): the size of the kernel for blurring.
                Defaults to 5.

        Returns:
            Self: image thresholded where its values are now pure 0 or 255
        """
        self.as_grayscale()
        if is_blur_enabled:
            im_arr = cv2.GaussianBlur(
                src=self.asarray, ksize=(blur_ksize, blur_ksize), sigmaX=0
            )
        else:
            im_arr = self.asarray

        self.asarray = np.asarray((im_arr > threshold_otsu(im_arr)) * 255).astype(
            np.uint8
        )
        return self

    def threshold_sauvola(self, window_size: int = 15, k: float = 0.2) -> Self:
        """Apply Sauvola thresholding.
        See https://scikit-image.org/docs/stable/auto_examples/segmentation/\
            plot_niblack_sauvola.html.

        As the input image must be a grayscale before applying any thresholding
        methods we convert the image to grayscale.

        Args:
            window_size (int, optional): the sauvola window size to apply on the
                image. Defaults to 15.
            k (float, optional): the sauvola k factor to apply to regulate the impact
                of the std. Defaults to 0.2.

        Returns:
            Self: image thresholded where its values are now pure 0 or 255
        """
        self.as_grayscale()
        im_arr = self.asarray
        self.asarray = np.asarray(
            (im_arr > threshold_sauvola(image=im_arr, window_size=window_size, k=k))
            * 255
        ).astype(np.uint8)
        return self

    def blur(self, kernel: tuple = (5, 5), iterations: int = 1) -> Self:
        """Blur the images

        Args:
            kernel (tuple, optional): blur kernel size. Defaults to (5, 5).
            iterations (int, optional): number of iterations. Defaults to 1.

        Returns:
            Self: the new image blurred
        """
        for _ in range(iterations):
            self.asarray = cv2.blur(src=self.asarray, ksize=kernel)
        return self

    def dilate(
        self,
        kernel: tuple = (5, 5),
        iterations: int = 1,
        dilate_black_pixels: bool = True,
    ) -> Self:
        """Dilate the image by making the black pixels expand in the image.
        The dilatation can be parametrize thanks to the kernel and iterations
        arguments.

        Args:
            kernel (tuple, optional): kernel to dilate. Defaults to (5, 5).
            iterations (int, optional): number of dilatation iterations. Defaults to 1.

        Returns:
            Self: image dilated
        """
        if dilate_black_pixels:
            self.asarray = (
                1
                - np.asarray(
                    cv2.dilate(
                        self.binaryrev(),
                        kernel=np.ones(kernel, np.uint8),
                        iterations=iterations,
                    ),
                    dtype=np.uint8,
                )
            ) * 255
        else:
            self.asarray = (
                np.asarray(
                    cv2.dilate(
                        self.binary(),
                        kernel=np.ones(kernel, np.uint8),
                        iterations=iterations,
                    ),
                    dtype=np.uint8,
                )
                * 255
            )
        return self

    def shift(self, shift: np.ndarray, mode: str = "constant") -> Self:
        """Shift the image doing a translation operation

        Args:
            shift (np.ndarray): Vector for translation
            mode (str, optional): Defaults to "contants"

        Returns:
            Self: image translated
        """
        self.asarray = scipy.ndimage.shift(
            input=self.asarray, shift=geo.Vector(shift).cv2_space_coords, mode=mode
        )
        return self

    def rotate(
        self, angle: float, is_degree: bool = True, reshape: bool = True
    ) -> Self:
        """Rotate the image

        Args:
            angle (float): angle to rotate the image
            is_degree (bool, optional): whether the angle is in degree or not.
                Defaults to True.
            reshape (bool, optional): scipy reshape option. Defaults to True.

        Returns:
            (Self): image rotated
        """
        if not is_degree:
            angle = np.rad2deg(angle)
        self.asarray = scipy.ndimage.rotate(
            input=self.asarray, angle=angle, reshape=reshape
        )
        return self

    def center_image_to_point(
        self, point: np.ndarray, mode: str = "constant"
    ) -> tuple[Self, np.ndarray]:
        """Shift the image so that the input point ends up in the middle of the
        new image

        Args:
            point (np.ndarray): point as (2,) shape numpy array
            mode (str, optional): scipy shift interpolation mode.
                Defaults to "constant".

        Returns:
            (tuple[Self, np.ndarray]): Self, translation Vector
        """
        shift_vector = self.center - point
        im = self.shift(shift=shift_vector, mode=mode)
        return im, shift_vector

    def center_image_to_segment(
        self, segment: np.ndarray, mode: str = "constant"
    ) -> tuple[Self, np.ndarray]:
        """Shift the image so that the line middle point ends up in the middle
        of the new image

        Args:
            segment (Segment): segment class object
            mode (str, optional): scipy mode for the translation.
                Defaults to "constant".

        Returns:
            (tuple[Self, np.ndarray]): Self, vector_shift
        """
        return self.center_image_to_point(
            point=geo.Segment(segment).centroid, mode=mode
        )

    def resize_fixed(
        self, dim: tuple[int, int], interpolation: int = cv2.INTER_AREA
    ) -> Self:
        """Resize the image using a fixed dimension well defined.
        This function can result in a distorted image.

        Args:
            dim (tuple[int, int]): a tuple with two integer, width, height.
            interpolation (int, optional): resize interpolation.
                Defaults to cv2.INTER_AREA.

        Returns:
            Self: image object
        """
        im = self.asarray

        if dim[0] < 0 and dim[1] < 0:  # check that the dim should be positive
            raise RuntimeError(
                f"The dim argument {dim} if not appropriate for" f"image resize"
            )

        # compute width or height
        _dim = list(dim)
        if _dim[0] <= 0:
            _dim[0] = int(self.width * (_dim[1] / self.height))
        if dim[1] <= 0:
            _dim[1] = int(self.height * (_dim[0] / self.width))

        self.asarray = cv2.resize(src=im, dsize=_dim, interpolation=interpolation)
        return self

    def resize(self, scale_pct: float, interpolation: int = cv2.INTER_AREA) -> Self:
        """Resize the image to a new size using a scaling percentage value that
        will be applied to all dimensions (width and height).
        Applying this method can not result in a distorted image.

        Args:
            scale_pct (float): scale to resize the image. A value 100 does not
                change the image. 200 double the image size.

        Returns:
            (Self): resized image
        """
        if scale_pct < 0:
            raise ValueError(
                f"The scale percent value {scale_pct} must be stricly positive"
            )
        if scale_pct > 5:
            raise ValueError(f"The scale percent value {scale_pct} is probably too big")

        if scale_pct == 1:
            return self

        width = int(self.width * scale_pct)
        height = int(self.height * scale_pct)
        dim = (width, height)
        return self.resize_fixed(dim=dim, interpolation=interpolation)

    def crop_image_horizontal(self, x0: int, y0: int, x1: int, y1: int) -> Self:
        """Crop an image

        Args:
            x0 (int): x coordinate of the first point
            y0 (int): y coordinate of the first point
            x1 (int): x coordinate of the second point
            y1 (int): y coordinate of the second point

        Returns:
            Self: image cropped
        """
        self.asarray = self.asarray[x0:x1, y0:y1]
        return self

    def crop_around_segment_horizontal(
        self,
        segment: np.ndarray,
        dim_crop_rect: tuple[int, int] = (-1, 100),
        default_extra_width: int = 75,
    ) -> tuple[Self, np.ndarray, float, np.ndarray]:
        """Crop around a specific segment in the image. This is done in three
        specific steps:
        1) shift image so that the middle of the segment is in the middle of the image
        2) rotate image by the angle of segment so that the segment becomes horizontal
        3) crop the image

        Args:
            segment (np.ndarray): Segment class object
            dim_crop_rect (tuple, optional): height, width. Defaults to 100.
            default_extra_width (int, optional): additional width for cropping.
                Defaults to 75.

        Returns:
            tuple[Self, np.ndarray, float]: _description_
        """
        width_crop_rect, height_crop_rect = dim_crop_rect
        im = self.copy()

        # center the image based on the middle of the line
        geo_segment = geo.Segment(segment)
        im, translation_vector = im.center_image_to_segment(segment=segment)

        if width_crop_rect == -1:
            # default the width for crop to be a bit more than line length
            width_crop_rect = int(geo_segment.length + default_extra_width)
        assert width_crop_rect > 0 and height_crop_rect > 0

        # rotate the image so that the line is horizontal
        angle = geo_segment.slope_angle(degree=True)
        im = im.rotate(angle=angle)

        # cropping
        im_crop = im.crop_image_horizontal(
            x0=int(im.center[1] - height_crop_rect / 2),
            y0=int(im.center[0] - width_crop_rect / 2),
            x1=int(im.center[1] + height_crop_rect / 2),
            y1=int(im.center[0] + width_crop_rect / 2),
        )

        crop_translation_vector = self.center - im_crop.center
        return im_crop, translation_vector, angle, crop_translation_vector
