"""
Geometry Transformer component
"""

import numpy as np
from numpy.typing import NDArray
from typing import Sequence
import cv2
import scipy.ndimage

from otary.image.base import BaseImage
import otary.geometry as geo
from otary.utils.tools import assert_transform_shift_vector


class GeometrizerImage:

    def __init__(self, base: BaseImage) -> None:
        self.base = base

    def shift(self, shift: NDArray, fill_value: Sequence[float] = (0.0,)) -> None:
        """Shift the image by performing a translation operation

        Args:
            shift (NDArray): Vector for translation
            border_fill_value (int | tuple[int, int, int], optional): value to fill the
                border of the image after the rotation in case reshape is True.
                Can be a tuple of 3 integers for RGB image or a single integer for
                grayscale image. Defaults to (0.0,) which is black.
        """
        vector_shift = assert_transform_shift_vector(vector=shift)
        shift_matrix = np.asarray(
            [[1.0, 0.0, vector_shift[0]], [0.0, 1.0, vector_shift[1]]],
            dtype=np.float32,
        )

        self.base.asarray = cv2.warpAffine(
            src=self.base.asarray,
            M=shift_matrix,
            dsize=(self.base.width, self.base.height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=fill_value,
        )  # type: ignore[call-overload]

    def __rotate_exact(
        self,
        angle: float,
        is_degree: bool = False,
        is_clockwise: bool = True,
        reshape: bool = True,
        border_fill_value: float = 0.0,
    ) -> None:
        """Rotate the image by a given angle.
        This method is more accurate than the rotate method but way slower
        (about 10 times slower).

        Args:
            angle (float): angle to rotate the image
            is_degree (bool, optional): whether the angle is in degree or not.
                If not it is considered to be in radians.
                Defaults to False which means radians.
            is_clockwise (bool, optional): whether the rotation is clockwise or
                counter-clockwise. Defaults to True.
            reshape (bool, optional): scipy reshape option. Defaults to True.
            border_fill_value (float, optional): value to fill the border of the image
                after the rotation in case reshape is True. Can only be a single
                integer. Does not support tuple of 3 integers for RGB image.
                Defaults to 0.0 which is black.
        """
        # pylint: disable=too-many-arguments, too-many-positional-arguments
        if not is_degree:
            angle = np.rad2deg(angle)
        if is_clockwise:
            # by default scipy rotate is counter-clockwise
            angle = -angle
        self.base.asarray = scipy.ndimage.rotate(
            input=self.base.asarray,
            angle=angle,
            reshape=reshape,
            cval=border_fill_value,
        )

    def rotate(
        self,
        angle: float,
        is_degree: bool = False,
        is_clockwise: bool = True,
        reshape: bool = True,
        fill_value: Sequence[float] = (0.0,),
        fast: bool = True,
    ) -> None:
        """Rotate the image by a given angle.

        For the rotation with reshape, meaning preserving the whole image,
        we used the code from the imutils library:
        https://github.com/PyImageSearch/imutils/blob/master/imutils/convenience.py#L41

        Args:
            angle (float): angle to rotate the image
            is_degree (bool, optional): whether the angle is in degree or not.
                If not it is considered to be in radians.
                Defaults to False which means radians.
            is_clockwise (bool, optional): whether the rotation is clockwise or
                counter-clockwise. Defaults to True.
            reshape (bool, optional): whether to preserve the original image or not.
                If True, the complete image is preserved hence the width and height
                of the rotated image are different than in the original image.
                Defaults to True.
            border_fill_value (Sequence[float], optional): value to
                fill the border of the image after the rotation in case reshape is True.
                Can be a tuple of 3 integers for RGB image or a single integer for
                grayscale image. Defaults to (0.0,) which is black.
        """
        # pylint: disable=too-many-arguments, too-many-positional-arguments
        # pylint: disable=too-many-locals
        if not fast:  # using scipy rotate which is slower than cv2
            border_fill_value_scalar = fill_value[0]
            if not isinstance(border_fill_value_scalar, float):
                raise ValueError(
                    f"The border_fill_value {border_fill_value_scalar} is not a valid "
                    "value. It must be a single integer when fast mode is off"
                )
            return self.__rotate_exact(
                angle=angle,
                is_degree=is_degree,
                is_clockwise=is_clockwise,
                reshape=reshape,
                border_fill_value=border_fill_value_scalar,
            )

        if not is_degree:
            angle = np.rad2deg(angle)
        if is_clockwise:
            angle = -angle

        h, w = self.base.asarray.shape[:2]
        center = (w / 2, h / 2)

        # Compute rotation matrix
        rotmat = cv2.getRotationMatrix2D(center, angle, 1.0)  # param angle in degree

        if reshape:
            # Compute new bounding dimensions
            cos_a = np.abs(rotmat[0, 0])
            sin_a = np.abs(rotmat[0, 1])
            new_w = int((h * sin_a) + (w * cos_a))
            new_h = int((h * cos_a) + (w * sin_a))
            w, h = new_w, new_h

            # Adjust the rotation matrix to shift the image center
            rotmat[0, 2] += (w / 2) - center[0]
            rotmat[1, 2] += (h / 2) - center[1]

        self.base.asarray = cv2.warpAffine(
            src=self.base.asarray,
            M=rotmat,
            dsize=(w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=fill_value,
        )  # type: ignore[call-overload]

    def center_to_point(self, point: NDArray) -> NDArray:
        """Shift the image so that the input point ends up in the middle of the
        new image

        Args:
            point (NDArray): point as (2,) shape numpy array

        Returns:
            (tuple[Self, NDArray]): Self, translation Vector
        """
        shift_vector = self.base.center - point
        self.shift(shift=shift_vector)
        return shift_vector

    def center_to_segment(self, segment: NDArray) -> NDArray:
        """Shift the image so that the segment middle point ends up in the middle
        of the new image

        Args:
            segment (NDArray): segment as numpy array of shape (2, 2)

        Returns:
            (tuple[Self, NDArray]): Self, vector_shift
        """
        return self.center_to_point(point=geo.Segment(segment).centroid)

    def restrict_rect_in_frame(self, rectangle: geo.Rectangle) -> geo.Rectangle:
        """Create a new rectangle that is contained within the image borders.
        If the input rectangle is outside the image, the returned rectangle is a
        image frame-fitted rectangle that preserve the same shape.

        Args:
            rectangle (geo.Rectangle): input rectangle

        Returns:
            geo.Rectangle: new rectangle
        """
        # rectangle boundaries
        xmin, xmax = rectangle.xmin, rectangle.xmax
        ymin, ymax = rectangle.ymin, rectangle.ymax

        # recalculate boundaries based on image shape
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(self.base.width, xmax)
        ymax = min(self.base.height, ymax)

        # recreate a rectangle with new coordinates
        rect_restricted = geo.Rectangle.from_topleft_bottomright(
            topleft=np.asarray([xmin, ymin]),
            bottomright=np.asarray([xmax, ymax]),
            is_cast_int=True,
        )
        return rect_restricted
