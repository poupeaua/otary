"""
Image Trasnformation module. it only contains advanced image transformation methods.
"""

from __future__ import annotations

from typing import Self, Sequence
from abc import ABC

import cv2
import numpy as np
import scipy.ndimage

# pylint: disable=no-name-in-module
from skimage.filters import threshold_sauvola

import src.geometry as geo
from src.image.base import BaseImage
from src.utils.tools import assert_transform_shift_vector


class TransformerImage(BaseImage, ABC):
    """Transformer images utility class"""

    # pylint: disable=too-many-public-methods

    def threshold_simple(self, thresh: int) -> Self:
        """Compute the image thesholded by a single value T.
        All pixels with value v <= T are turned black and those with value v > T are
        turned white.

        Args:
            threshold_value (int): value to separate the black from the white pixels.

        Returns:
            Self: new image thresholded
        """
        self.as_grayscale()
        self.asarray = np.array((self.asarray > thresh) * 255, dtype=np.uint8)
        return self

    def threshold_adaptative(self) -> Self:
        """Apply adaptive thresholding.

        A median blur is applied before for better thresholding results.
        See https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html.

        As the input image must be a grayscale before applying any thresholding
        methods we convert the image to grayscale.

        Returns:
            Self: image thresholded where its values are now pure 0 or 255
        """
        self.as_grayscale()
        binary = cv2.adaptiveThreshold(
            self.asarray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        self.asarray = binary
        return self

    def threshold_otsu(self) -> Self:
        """Apply Ostu thresholding.

        A gaussian blur is applied before for better thresholding results.
        See https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html.

        As the input image must be a grayscale before applying any thresholding
        methods we convert the image to grayscale.

        Returns:
            Self: image thresholded where its values are now pure 0 or 255
        """
        self.as_grayscale()
        self.blur(method="gaussian", kernel=(3, 3), sigmax=0)
        _, binary = cv2.threshold(
            self.asarray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        self.asarray = binary
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
        self.asarray = np.array(
            (im_arr > threshold_sauvola(image=im_arr, window_size=window_size, k=k))
            * 255,
            dtype=np.uint8,
        )
        return self

    def binary(self, method: str = "sauvola") -> np.ndarray:
        """Binary representation of the image with values that can be only 0 or 1.
        The value 0 is now 0 and value of 255 are now 1. Black is 0 and white is 1.
        We can also talk about the mask of the image to refer to the binary
        representation of it.

        The sauvola is generally the best binarization method however it is
        way slower than the others methods. The adaptative or otsu method are the best
        method in terms of speed and quality.

        Args:
            method (str, optional): the binarization method to apply.
                Must be in ["adaptative", "otsu", "sauvola"].
                Defaults to "adaptative".

        Returns:
            np.ndarray: array where its inner values are 0 or 1
        """
        valid_binarization_methods = ["adaptative", "otsu", "sauvola"]

        if method == "adaptative":
            return self.threshold_adaptative().asarray_norm
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
                Defaults to "adaptative".

        Returns:
            np.ndarray: array where its inner values are 0 or 1
        """
        return 1 - self.binary(method=method)

    def rev(self) -> Self:
        """Reverse the image colors. Each pixel color value V becomes |V - 255|.

        Applied on a grayscale image the black pixel becomes white and the
        white pixels become black.
        """
        self.asarray = np.abs(self.asarray.astype(np.int16) - 255).astype(np.uint8)
        return self

    def blur(
        self,
        kernel: tuple = (5, 5),
        iterations: int = 1,
        method: str = "average",
        sigmax: float = 0,
    ) -> Self:
        """Blur the images

        Args:
            kernel (tuple, optional): blur kernel size. Defaults to (5, 5).
            iterations (int, optional): number of iterations. Defaults to 1.
            method (str, optional): blur method.
                Must be in ["average", "median", "gaussian", "bilateral"].
                Defaults to "average".

        Returns:
            Self: the new image blurred
        """
        blur_valid_methods = ["average", "median", "gaussian", "bilateral"]
        if method not in blur_valid_methods:
            raise ValueError(
                f"The blur method {method} is not a valid method. "
                f"A valid method must be in {blur_valid_methods}"
            )
        for _ in range(iterations):
            if method == "average":
                self.asarray = cv2.blur(src=self.asarray, ksize=kernel)
            elif method == "median":
                self.asarray = cv2.medianBlur(src=self.asarray, ksize=kernel[0])
            elif method == "gaussian":
                self.asarray = cv2.GaussianBlur(
                    src=self.asarray, ksize=kernel, sigmaX=sigmax
                )
            elif method == "bilateral":
                self.asarray = cv2.bilateralFilter(
                    src=self.asarray, d=kernel[0], sigmaColor=75, sigmaSpace=75
                )
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
            dilate_black_pixels (bool, optional): whether to dilate black pixels or not

        Returns:
            Self: image dilated
        """
        if iterations == 0:
            return self

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

    def erode(
        self,
        kernel: tuple = (5, 5),
        iterations: int = 1,
        erode_black_pixels: bool = True,
    ) -> Self:
        """Erode the image by making the black pixels shrink in the image.
        The anti-dilatation can be parametrize thanks to the kernel and iterations
        arguments.

        Args:
            kernel (tuple, optional): kernel to erode. Defaults to (5, 5).
            iterations (int, optional): number of iterations. Defaults to 1.
            erode_black_pixels (bool, optional): whether to erode black pixels or not

        Returns:
            Self: image eroded
        """
        if iterations == 0:
            return self

        if erode_black_pixels:
            self.asarray = (
                1
                - np.asarray(
                    cv2.erode(
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
                    cv2.erode(
                        self.binary(),
                        kernel=np.ones(kernel, np.uint8),
                        iterations=iterations,
                    ),
                    dtype=np.uint8,
                )
                * 255
            )
        return self

    def shift(
        self, shift: np.ndarray, border_fill_value: Sequence[float] = (0.0,)
    ) -> Self:
        """Shift the image doing a translation operation

        Args:
            shift (np.ndarray): Vector for translation
            border_fill_value (int | tuple[int, int, int], optional): value to fill the
                border of the image after the rotation in case reshape is True.
                Can be a tuple of 3 integers for RGB image or a single integer for
                grayscale image. Defaults to (0.0,) which is black.

        Returns:
            Self: image translated
        """
        vector_shift = assert_transform_shift_vector(vector=shift)
        shift_matrix = np.asarray(
            [[1.0, 0.0, vector_shift[0]], [0.0, 1.0, vector_shift[1]]],
            dtype=np.float32,
        )

        self.asarray = cv2.warpAffine(
            src=self.asarray,
            M=shift_matrix,
            dsize=(self.width, self.height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_fill_value,
        )  # type: ignore[call-overload]
        return self

    def __rotate_exact(
        self,
        angle: float,
        is_degree: bool = False,
        is_clockwise: bool = True,
        reshape: bool = True,
        border_fill_value: float = 0.0,
    ) -> Self:
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

        Returns:
            (Self): image rotated
        """
        # pylint: disable=too-many-arguments, too-many-positional-arguments
        if not is_degree:
            angle = np.rad2deg(angle)
        if is_clockwise:
            # by default scipy rotate is counter-clockwise
            angle = -angle
        self.asarray = scipy.ndimage.rotate(
            input=self.asarray, angle=angle, reshape=reshape, cval=border_fill_value
        )
        return self

    def rotate(
        self,
        angle: float,
        is_degree: bool = False,
        is_clockwise: bool = True,
        reshape: bool = True,
        border_fill_value: Sequence[float] = (0.0,),
        fast: bool = True,
    ) -> Self:
        """Rotate the image by a given angle.

        For the rotation with reshape, meaning preserveing the whole image,
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

        Returns:
            (Self): image rotated
        """
        # pylint: disable=too-many-arguments, too-many-positional-arguments
        # pylint: disable=too-many-locals
        if not fast:  # using scipy rotate which is slower than cv2
            border_fill_value_scalar = border_fill_value[0]
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

        h, w = self.asarray.shape[:2]
        center = (w / 2, h / 2)

        # Compute rotation matrix
        rotmat = cv2.getRotationMatrix2D(center, angle, 1.0)

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

        self.asarray = cv2.warpAffine(
            src=self.asarray,
            M=rotmat,
            dsize=(w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_fill_value,
        )  # type: ignore[call-overload]

        return self

    def center_image_to_point(self, point: np.ndarray) -> tuple[Self, np.ndarray]:
        """Shift the image so that the input point ends up in the middle of the
        new image

        Args:
            point (np.ndarray): point as (2,) shape numpy array

        Returns:
            (tuple[Self, np.ndarray]): Self, translation Vector
        """
        shift_vector = self.center - point
        self.shift(shift=shift_vector)
        return self, shift_vector

    def center_image_to_segment(self, segment: np.ndarray) -> tuple[Self, np.ndarray]:
        """Shift the image so that the segment middle point ends up in the middle
        of the new image

        Args:
            segment (np.ndarray): segment as numpy array of shape (2, 2)

        Returns:
            (tuple[Self, np.ndarray]): Self, vector_shift
        """
        return self.center_image_to_point(point=geo.Segment(segment).centroid)

    def resize_fixed(
        self,
        dim: tuple[int, int],
        interpolation: int = cv2.INTER_AREA,
        copy: bool = False,
    ) -> Self:
        """Resize the image using a fixed dimension well defined.
        This function can result in a distorted image if the ratio between
        width and height is different in the original and the new image.

        If the dim argument has a negative value in height or width, then
        a proportional ratio is applied based on the one of the two dimension given.

        Args:
            dim (tuple[int, int]): a tuple with two integers in the following order
                (width, height).
            interpolation (int, optional): resize interpolation.
                Defaults to cv2.INTER_AREA.

        Returns:
            Self: image object
        """
        if dim[0] < 0 and dim[1] < 0:  # check that the dim is positive
            raise ValueError(f"The dim argument {dim} has two negative values.")

        _dim = list(dim)

        # compute width or height if needed
        if _dim[1] <= 0:
            _dim[1] = int(self.height * (_dim[0] / self.width))
        if dim[0] <= 0:
            _dim[0] = int(self.width * (_dim[1] / self.height))

        result = cv2.resize(src=self.asarray, dsize=_dim, interpolation=interpolation)

        if copy:
            return type(self)(image=result)

        self.asarray = result
        return self

    def resize(
        self, factor: float, interpolation: int = cv2.INTER_AREA, copy: bool = False
    ) -> Self:
        """Resize the image to a new size using a scaling factor value that
        will be applied to all dimensions (width and height).

        Applying this method can not result in a distorted image.

        Args:
            factor (float): factor in [0, 5] to resize the image.
                A value of 1 does not change the image.
                A value of 2 doubles the image size.
                A maximum value of 5 is set to avoid accidentally producing a gigantic
                image.

        Returns:
            (Self): resized image
        """
        if factor == 1:
            return self

        if factor < 0:
            raise ValueError(
                f"The resize factor value {factor} must be stricly positive"
            )

        max_scale_pct = 5
        if factor > max_scale_pct:
            raise ValueError(f"The resize factor value {factor} is probably too big")

        width = int(self.width * factor)
        height = int(self.height * factor)
        dim = (width, height)
        return self.resize_fixed(dim=dim, interpolation=interpolation, copy=copy)

    def __crop_with_padding(self, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        # Output size
        crop_width = x1 - x0
        crop_height = y1 - y0

        # Initialize output with black (zeros), same dtype and channel count
        channels = 1 if self.is_gray else self.asarray.shape[2]
        output_shape = (
            (crop_height, crop_width)
            if channels == 1
            else (crop_height, crop_width, channels)
        )
        result = np.zeros(output_shape, dtype=np.uint8)

        # Compute the intersection of crop with image bounds
        ix0 = max(x0, 0)
        iy0 = max(y0, 0)
        ix1 = min(x1, self.width)
        iy1 = min(y1, self.height)

        # Compute corresponding position in output
        ox0 = ix0 - x0
        oy0 = iy0 - y0
        ox1 = ox0 + (ix1 - ix0)
        oy1 = oy0 + (iy1 - iy0)

        # Copy the valid region
        result[oy0:oy1, ox0:ox1] = self.asarray[iy0:iy1, ix0:ix1]
        return result

    def __crop_with_clipping(self, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
        """Crop the image. A straight axis-aligned rectangle is used for cropping.
        This function inputs represents the top-left and bottom-right points.

        This method does not provide a way to extract a rotated rectangle or a
        different shape from the image.

        Remember that in this library the x coordinates represent the y coordinates of
        the image array (horizontal axis of the image).
        The array representation is always rows then columns.
        In this library this is the contrary like in opencv.

        Args:
            x0 (int): x coordinate of the first point
            y0 (int): y coordinate of the first point
            x1 (int): x coordinate of the second point
            y1 (int): y coordinate of the second point

        Returns:
            Self: image cropped
        """
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        if x0 >= self.width or y0 >= self.height or x1 <= 0 or y1 <= 0:
            raise ValueError(
                f"The coordinates ({x0}, {y0}, {x1}, {y1}) are out of the image "
                f"boundaries (width={self.width}, height={self.height})"
            )

        def clip(value: int, min_value: int, max_value: int) -> int:
            return max(min_value, min(value, max_value))

        x0 = clip(x0, 0, self.width - 1)
        y0 = clip(y0, 0, self.height - 1)
        x1 = clip(x1, 0, self.width - 1)
        y1 = clip(y1, 0, self.height - 1)

        result = self.asarray[int(y0) : int(y1) + 1, int(x0) : int(x1) + 1]
        return result

    def crop(
        self,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        clip: bool = True,
        pad: bool = False,
        copy: bool = False,
    ) -> Self:
        if (clip and pad) or (not clip and not pad):
            raise ValueError(f"When cropping clip and pad cannot be both {clip}")

        if clip:
            array_crop = self.__crop_with_clipping(x0=x0, y0=y0, x1=x1, y1=y1)
        if pad:
            array_crop = self.__crop_with_padding(x0=x0, y0=y0, x1=x1, y1=y1)

        if copy:
            return type(self)(image=array_crop)

        self.asarray = array_crop
        return self

    def crop_from_topleft(self, topleft: np.ndarray, width: int, height: int) -> Self:
        """Crop the image from a rectangle defined by its top-left point, its width and
        its height.

        Args:
            topleft (np.ndarray): (x, y) coordinates of the top-left point
            width (int): width of the rectangle to crop
            height (int): height of the rectangle to crop

        Returns:
            Self: image cropped
        """
        return self.crop(
            x0=topleft[0],
            y0=topleft[1],
            x1=topleft[0] + width - 1,
            y1=topleft[1] + height - 1,
        )

    def crop_from_axis_aligned_bbox(self, bbox: geo.Rectangle) -> Self:
        """Crop the image from an Axis-Aligned Bounding Box (AABB)

        Args:
            bbox (geo.Rectangle): axis-aligned bounding box

        Returns:
            Self: cropped image
        """
        topleft = np.asarray([bbox.xmin, bbox.ymin])
        height = int(bbox.ymax - bbox.ymin + 1)
        width = int(bbox.xmax - bbox.xmin + 1)
        return self.crop_from_topleft(topleft=topleft, width=width, height=height)

    def crop_around_segment_horizontal(
        self,
        segment: np.ndarray,
        dim_crop_rect: tuple[int, int] = (-1, 100),
        added_width: int = 75,
    ) -> tuple[Self, np.ndarray, float, np.ndarray]:
        """Crop around a specific segment in the image. This is done in three
        specific steps:
        1) shift image so that the middle of the segment is in the middle of the image
        2) rotate image by the angle of segment so that the segment becomes horizontal
        3) crop the image

        Args:
            segment (np.ndarray): segment as numpy array of shape (2, 2).
            dim_crop_rect (tuple, optional): represents (width, height).
                Defaults to heigth of 100 and width of -1 which means
                that the width is automatically computed based on the length of
                the segment.
            added_width (int, optional): additional width for cropping.
                Half of the added_width is added to each side of the segment.
                Defaults to 75.

        Returns:
            tuple[Self, np.ndarray, float, np.ndarray]: returns in the following order:
                1) the cropped image
                2) the translation vector used to center the image
                3) the angle of rotation applied to the image
                4) the translation vector used to crop the image
        """
        width_crop_rect, height_crop_rect = dim_crop_rect
        im = self.copy()

        # center the image based on the middle of the line
        geo_segment = geo.Segment(segment)
        im, translation_vector = im.center_image_to_segment(segment=segment)

        if width_crop_rect == -1:
            # default the width for crop to be a bit more than line length
            width_crop_rect = int(geo_segment.length)
        width_crop_rect += added_width
        assert width_crop_rect > 0 and height_crop_rect > 0

        # rotate the image so that the line is horizontal
        angle = geo_segment.slope_angle(is_cv2=True)
        im = im.rotate(angle=angle)

        # cropping
        im_crop = im.crop(
            x0=int(im.center[0] - width_crop_rect / 2),
            y0=int(im.center[1] - height_crop_rect / 2),
            x1=int(im.center[0] + width_crop_rect / 2),
            y1=int(im.center[1] + height_crop_rect / 2),
        )

        crop_translation_vector = self.center - im_crop.center
        return im_crop, translation_vector, angle, crop_translation_vector

    def crop_around_segment_horizontal_faster(
        self,
        segment: np.ndarray,
        dim_crop_rect: tuple[int, int] = (-1, 100),
        added_width: int = 75,
    ) -> Self:
        """Crop around a specific segment in the image.
        This method is generally faster especially for large images.

        Here is a comparison of the total time taken for cropping with the two methods
        with a loop over 1000 iterations:

        | Image dimension | Crop v1 | Crop faster |
        |-----------------|---------|-------------|
        | 1224 x 946      | 2.0s    | 0.25s       |
        | 2448 x 1892     | 4.51s   | 0.25s       |
        | 4896 x 3784     | 23.2s   | 0.25s       |

        Args:
            segment (np.ndarray): segment as numpy array of shape (2, 2).
            dim_crop_rect (tuple, optional): represents (width, height).
                Defaults to heigth of 100 and width of -1 which means
                that the width is automatically computed based on the length of
                the segment.
            added_width (int, optional): additional width for cropping.
                Half of the added_width is added to each side of the segment.
                Defaults to 75.

        Returns:
            Self: cropped image around the segment
        """
        width_crop_rect, height_crop_rect = dim_crop_rect
        geo_segment = geo.Segment(segment)
        angle = geo_segment.slope_angle(is_cv2=True)

        if width_crop_rect == -1:
            # default the width for crop to be a bit more than line length
            width_crop_rect = int(geo_segment.length)
        width_crop_rect += added_width
        assert width_crop_rect > 0 and height_crop_rect > 0

        x_extra = abs(added_width / 2 * np.cos(angle))
        y_extra = abs(added_width / 2 * np.sin(angle))

        im = self.crop(
            x0=geo_segment.xmin - x_extra,
            y0=geo_segment.ymin - y_extra,
            x1=geo_segment.xmax + x_extra,
            y1=geo_segment.ymax + y_extra,
            pad=True,
            clip=False,
            copy=True,  # copy the image after cropping for very fast performance
        )

        # rotate the image so that the line is horizontal
        im.rotate(angle=angle)

        # cropping
        im.crop(
            x0=int(im.center[0] - width_crop_rect / 2),
            y0=int(im.center[1] - height_crop_rect / 2),
            x1=int(im.center[0] + width_crop_rect / 2),
            y1=int(im.center[1] + height_crop_rect / 2),
            clip=True,
            pad=False,
            copy=False,
        )

        return im
