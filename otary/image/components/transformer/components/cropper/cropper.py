"""
Cropper Transformer component
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

import otary.geometry as geo
from otary.image.base import BaseImage

if TYPE_CHECKING:
    from otary.image import Image


class CropperImage:

    def __init__(self, base: BaseImage) -> None:
        self.base = base

    def __crop_with_padding(
        self, x0: int, y0: int, x1: int, y1: int, pad_value: int = 0
    ) -> NDArray:
        """Crop the image in a straight axis-aligned rectangle way given
        by the top-left point [x0, y0] and the bottom-right point [x1, y1].

        This method is specific to crop with padding meaning that if the
        coordinates are out of the image bounds, the padding is added to the
        output cropped image with the pad value parameter, black by default.

        Args:
            x0 (int): x coordinate of the top-left point
            y0 (int): y coordinate of the top-left point
            x1 (int): x coordinate of the bottom-right point
            y1 (int): y coordinate of the bottom-right point
            pad_value (int, optional): pad fill value. Defaults to 0.

        Returns:
            NDArray: output cropped image
        """
        # pylint: disable=too-many-locals
        # pylint: disable=too-many-arguments, too-many-positional-arguments
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        # Output size
        crop_width = x1 - x0
        crop_height = y1 - y0

        # Initialize output with black (zeros), same dtype and channel count
        channels = 1 if self.base.is_gray else self.base.asarray.shape[2]
        output_shape = (
            (crop_height, crop_width)
            if channels == 1
            else (crop_height, crop_width, channels)
        )
        result = np.full(shape=output_shape, fill_value=pad_value, dtype=np.uint8)

        # Compute the intersection of crop with image bounds
        ix0 = max(x0, 0)
        iy0 = max(y0, 0)
        ix1 = min(x1, self.base.width)
        iy1 = min(y1, self.base.height)

        # Compute corresponding position in output
        ox0 = ix0 - x0
        oy0 = iy0 - y0
        ox1 = ox0 + (ix1 - ix0)
        oy1 = oy0 + (iy1 - iy0)

        # Copy the valid region
        result[oy0:oy1, ox0:ox1] = self.base.asarray[iy0:iy1, ix0:ix1]

        return result

    def __crop_with_clipping(self, x0: int, y0: int, x1: int, y1: int) -> NDArray:
        """Crop the image in a straight axis-aligned rectangle way given
        by the top-left point [x0, y0] and the bottom-right point [x1, y1].

        Crop by clipping meaning that if the coordinates are out of the image
        bounds the output is only the part of the image that is in the bounds.

        Args:
            x0 (int): x coordinate of the top-left point
            y0 (int): y coordinate of the top-left point
            x1 (int): x coordinate of the bottom-right point
            y1 (int): y coordinate of the bottom-right point

        Returns:
            Self: image cropped
        """
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        if x0 >= self.base.width or y0 >= self.base.height or x1 <= 0 or y1 <= 0:
            raise ValueError(
                f"The coordinates ({x0}, {y0}, {x1}, {y1}) are out of the image "
                f"boundaries (width={self.base.width}, height={self.base.height}). "
                "No crop is possible."
            )

        def clip(value: int, min_value: int, max_value: int) -> int:
            return int(max(min_value, min(value, max_value)))

        x0 = clip(x0, 0, self.base.width)
        y0 = clip(y0, 0, self.base.height)
        x1 = clip(x1, 0, self.base.width)
        y1 = clip(y1, 0, self.base.height)

        result = self.base.asarray[y0:y1, x0:x1]
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
        extra_border_size: int = 0,
        pad_value: int = 0,
    ) -> Optional[Image]:
        """Crop the image in a straight axis-aligned rectangle way given
        by the top-left point [x0, y0] and the bottom-right point [x1, y1]

        This function inputs represents the top-left and bottom-right points.
        This method does not provide a way to extract a rotated rectangle or a
        different shape from the image.

        Remember that in this library the x coordinates represent the y coordinates of
        the image array (horizontal axis of the image).
        The array representation is always rows then columns.
        In this library this is the contrary like in opencv.

        Args:
            x0 (int): top-left x coordinate
            y0 (int): top-left y coordinate
            x1 (int): bottom-right x coordinate
            y1 (int): bottom-right y coordinate
            clip (bool, optional): whether to clip or not. Defaults to True.
            pad (bool, optional): whether to pad or not. Defaults to False.
            copy (bool, optional): whether to copy or not. Defaults to False.
            extra_border_size (int, optional): extra border size to add to the crop
                in the x and y directions. Defaults to 0 which means no extra border.
            pad_value (int, optional): pad fill value. Defaults to 0.

        Returns:
            Optional[Image]: cropped image if copy=True else None
        """
        # pylint: disable=too-many-arguments, too-many-positional-arguments
        if (clip and pad) or (not clip and not pad):
            raise ValueError(f"Parameters clip and pad cannot be both {clip}")

        if clip and not pad:
            array_crop = self.__crop_with_clipping(
                x0=x0 - extra_border_size,
                y0=y0 - extra_border_size,
                x1=x1 + extra_border_size,
                y1=y1 + extra_border_size,
            )
        else:  # pad and not clip:
            array_crop = self.__crop_with_padding(
                x0=x0 - extra_border_size,
                y0=y0 - extra_border_size,
                x1=x1 + extra_border_size,
                y1=y1 + extra_border_size,
                pad_value=pad_value,
            )

        if copy:
            # really important feature to allow new image from original
            # without the user doing image.copy().crop()
            # which would be much more expensive if the image is large
            # this is why the output of the methods is Optional[Image] not None
            from otary.image import Image

            return Image(image=array_crop)

        self.base.asarray = array_crop
        return None

    def crop_from_topleft(
        self,
        topleft: np.ndarray,
        width: int,
        height: int,
        clip: bool = True,
        pad: bool = False,
        copy: bool = False,
        extra_border_size: int = 0,
        pad_value: int = 0,
    ) -> Optional[Image]:
        """Crop the image from a rectangle defined by its top-left point, its width and
        its height.

        Args:
            topleft (np.ndarray): (x, y) coordinates of the top-left point
            width (int): width of the rectangle to crop
            height (int): height of the rectangle to crop
            clip (bool, optional): whether to clip or not. Defaults to True.
            pad (bool, optional): whether to pad or not. Defaults to False.
            copy (bool, optional): whether to copy or not. Defaults to False.
            extra_border_size (int, optional): extra border size to add to the crop
                in the x and y directions. Defaults to 0 which means no extra border.
            pad_value (int, optional): pad fill value. Defaults to 0.

        Returns:
            Optional[Image]: image cropped if copy=True else None
        """
        # pylint: disable=too-many-arguments, too-many-positional-arguments
        return self.crop(
            x0=topleft[0],
            y0=topleft[1],
            x1=topleft[0] + width,
            y1=topleft[1] + height,
            clip=clip,
            pad=pad,
            copy=copy,
            extra_border_size=extra_border_size,
            pad_value=pad_value,
        )

    def crop_from_center(
        self,
        center: NDArray,
        width: int,
        height: int,
        clip: bool = True,
        pad: bool = False,
        copy: bool = False,
        extra_border_size: int = 0,
        pad_value: int = 0,
    ) -> Optional[Image]:
        """Crop the image from a rectangle defined by its center point, its width and
        its height.

        Args:
            center (NDArray): (x, y) coordinates of the center point
            width (int): width of the rectangle to crop
            height (int): height of the rectangle to crop
            clip (bool, optional): whether to clip or not. Defaults to True.
            pad (bool, optional): whether to pad or not. Defaults to False.
            copy (bool, optional): whether to copy or not. Defaults to False.
            extra_border_size (int, optional): extra border size to add to the crop
                in the x and y directions. Defaults to 0 which means no extra border.
            pad_value (int, optional): pad fill value. Defaults to 0.

        Returns:
            Optional[Image]: image cropped if copy=True else None
        """
        # pylint: disable=too-many-arguments, too-many-positional-arguments
        return self.crop_from_topleft(
            topleft=center - np.array([width / 2, height / 2]),
            width=width,
            height=height,
            clip=clip,
            pad=pad,
            copy=copy,
            extra_border_size=extra_border_size,
            pad_value=pad_value,
        )

    def crop_from_polygon(
        self,
        polygon: geo.Polygon,
        copy: bool = False,
        clip: bool = True,
        pad: bool = False,
        extra_border_size: int = 0,
        pad_value: int = 0,
    ) -> Optional[Image]:
        """Crop the image from a polygon"""
        # pylint: disable=too-many-arguments, too-many-positional-arguments
        return self.crop(
            x0=int(polygon.xmin),
            y0=int(polygon.ymin),
            x1=int(polygon.xmax),
            y1=int(polygon.ymax),
            copy=copy,
            clip=clip,
            pad=pad,
            extra_border_size=extra_border_size,
            pad_value=pad_value,
        )

    def crop_from_linear_spline(
        self,
        spline: geo.LinearSpline,
        copy: bool = False,
        clip: bool = True,
        pad: bool = False,
        extra_border_size: int = 0,
        pad_value: int = 0,
    ) -> Optional[Image]:
        """Crop the image from a linear spline"""
        # pylint: disable=too-many-arguments, too-many-positional-arguments
        return self.crop(
            x0=int(spline.xmin),
            y0=int(spline.ymin),
            x1=int(spline.xmax),
            y1=int(spline.ymax),
            copy=copy,
            clip=clip,
            pad=pad,
            extra_border_size=extra_border_size,
            pad_value=pad_value,
        )

    def crop_from_axis_aligned_bbox(
        self,
        bbox: geo.Rectangle,
        clip: bool = True,
        pad: bool = False,
        copy: bool = False,
        extra_border_size: int = 0,
        pad_value: int = 0,
    ) -> Optional[Image]:
        """Crop the image from an Axis-Aligned Bounding Box (AABB).
        Inclusive crops which means that the cropped image will have
        width and height equal to the width and height of the AABB.

        Args:
            bbox (geo.Rectangle): axis-aligned bounding box
            clip (bool, optional): whether to clip or not. Defaults to True.
            pad (bool, optional): whether to pad or not. Defaults to False.
            copy (bool, optional): whether to copy or not. Defaults to False.
            extra_border_size (int, optional): extra border size to add to the crop
                in the x and y directions. Defaults to 0 which means no extra border.
            pad_value (int, optional): pad fill value. Defaults to 0.

        Returns:
            Optional[Image]: cropped image if copy=True else None
        """
        # pylint: disable=too-many-arguments, too-many-positional-arguments
        assert bbox.is_axis_aligned
        topleft = np.asarray([bbox.xmin, bbox.ymin])
        height = int(bbox.ymax - bbox.ymin + 1)
        width = int(bbox.xmax - bbox.xmin + 1)
        return self.crop_from_topleft(
            topleft=topleft,
            width=width,
            height=height,
            clip=clip,
            pad=pad,
            copy=copy,
            extra_border_size=extra_border_size,
            pad_value=pad_value,
        )
