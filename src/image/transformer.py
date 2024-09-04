"""
Image manipulation module
"""

from __future__ import annotations

from typing import Self

from abc import ABC
import cv2
import numpy as np
import scipy.ndimage
from skimage.filters import threshold_sauvola, threshold_otsu

import src.geometry as geo
from src.image.base import BaseImage


class TransformerImage(BaseImage, ABC):
    """Transform images utility class"""

    def binary(self) -> np.ndarray:
        """Binary representation of the image with values that can be only 0 or 1.
        The value 0 is now 0 and value of 255 are now 1. Black is 0 and white is 1.

        Returns:
            np.ndarray: array where its inner values are 0 or 1
        """
        return self.threshold_otsu().asarray_norm.astype(np.uint8)

    def binaryrev(self) -> np.ndarray:
        """Reversed binary representation of the image.
        The value 0 is now 1 and value of 255 are now 0. Black is 1 and white is 0.

        Returns:
            np.ndarray: array where its inner values are 0 or 1
        """
        return 1 - self.binary()

    def threshold_otsu(self, is_blur_enabled: bool = True, blur_ksize: int = 5) -> Self:
        """Apply Ostu thresholding. A blur is applied before for better masking results.
        See https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html.

        As the input image must be a grayscale before applying any thresholding
        methods we convert the image to grayscale.

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

        self.asarray = (im_arr > threshold_otsu(im_arr)) * 255
        return self

    def threshold_sauvola(self) -> Self:
        """Apply Sauvola thresholding.
        See https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_niblack_sauvola.html.

        As the input image must be a grayscale before applying any thresholding
        methods we convert the image to grayscale.

        Returns:
            Self: image thresholded where its values are now pure 0 or 255
        """
        self.as_grayscale()
        im_arr = self.asarray
        self.asarray = (im_arr > threshold_sauvola(im_arr)) * 255
        return self

    def dilate(self, kernel: tuple = (5, 5), iterations: int = 1) -> Self:
        """Dilate the image by making the black pixels expand in the image.
        The dilatation can be parametrize thanks to the kernel and iterations
        arguments.

        Args:
            kernel (tuple, optional): kernel to dilate. Defaults to (5, 5).
            iterations (int, optional): number of dilatation iterations. Defaults to 1.

        Returns:
            Self: image dilated
        """
        self.asarray = (
            1
            - cv2.dilate(
                self.binaryrev(),
                kernel=np.ones(kernel, np.uint8),
                iterations=iterations,
            )
        ) * 255
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

    def resize(self, scale_percent: float, interpolation: int = cv2.INTER_AREA) -> Self:
        """Resize the image to a new size using a scaling percentage value that
        will be applied to all dimensions (width and height).
        Applying this method can not result in a distorted image.

        Args:
            scale_percent (float): scale to resize the image. A value 100 does not
                change the image. 200 double the image size.

        Returns:
            (Self): resized image
        """
        if scale_percent == 100:
            return self

        width = int(self.width * scale_percent / 100)
        height = int(self.height * scale_percent / 100)
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
