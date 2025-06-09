"""
Image module for image processing.

The architecture design of this class follow the composition design pattern rather
than inheritance. This is because the Image has got a "has-a" relationship with
the other classes not a "is-a" relationship.
"""

from __future__ import annotations

from typing import Self, Optional, Literal, Sequence

import numpy as np
from numpy.typing import NDArray
import cv2
import PIL.Image as ImagePIL
import pymupdf

import src.geometry as geo
from src.geometry.discrete.linear.entity import LinearEntity
from src.cv.ocr.output import OcrSingleOutput
from src.image.base import BaseImage
from src.image.components import ReaderImage, WriterImage, DrawerImage, TransformerImage
from src.image.components.transformer.components.binarizer.binarizer import (
    BinarizationMethods,
)
from src.image.components.transformer.components.morphologyzer.morphologyzer import (
    BlurMethods,
)
from src.image.components.drawer import (
    PointsRender,
    CirclesRender,
    SegmentsRender,
    LinearSplinesRender,
    PolygonsRender,
    OcrSingleOutputRender,
)

ScoreDistanceFromCenterMethods = Literal["linear", "gaussian"]


class Image:
    """
    Image core class. It groups all the methods available from composition
    from all the image classes.
    """

    reader = ReaderImage()

    def __init__(self, image: NDArray) -> None:
        self._base = BaseImage(image=image)
        self.drawer = DrawerImage(base=self._base)
        self.writer = WriterImage(base=self._base)
        self.transformer = TransformerImage(base=self._base)

    # -------------------------------- CLASS METHODS ----------------------------------

    @classmethod
    def from_fillvalue(cls, value: int = 255, shape: tuple = (128, 128, 3)) -> Image:
        return cls(image=cls.reader.from_fillvalue(value=value, shape=shape))

    @classmethod
    def from_file(
        cls, filepath: str, as_grayscale: bool = False, resolution: Optional[int] = None
    ) -> Image:
        return cls(
            cls.reader.from_file(
                filepath=filepath, as_grayscale=as_grayscale, resolution=resolution
            )
        )

    @classmethod
    def from_pdf(
        cls,
        filepath: str,
        as_grayscale: bool = False,
        page_nb: int = 0,
        resolution: Optional[int] = None,
        clip_pct: Optional[pymupdf.Rect] = None,
    ) -> Image:
        return cls(
            cls.reader.from_pdf(
                filepath=filepath,
                as_grayscale=as_grayscale,
                page_nb=page_nb,
                resolution=resolution,
                clip_pct=clip_pct,
            )
        )

    # ---------------------------------- PROPERTIES -----------------------------------

    @property
    def asarray(self) -> np.ndarray:
        return self._base.asarray

    @asarray.setter
    def asarray(self, value: NDArray) -> None:
        self._base.asarray = value

    @property
    def asarray_binary(self) -> NDArray:
        return self._base.asarray_binary

    @property
    def width(self) -> int:
        return self._base.width

    @property
    def height(self) -> int:
        return self._base.height

    @property
    def center(self) -> NDArray[np.int16]:
        return self._base.center

    @property
    def area(self) -> int:
        return self._base.area

    @property
    def shape_array(self) -> tuple[int, int, int]:
        return self._base.shape_array

    @property
    def is_gray(self) -> bool:
        return self._base.is_gray

    @property
    def norm_side_length(self) -> int:
        return self._base.norm_side_length

    @property
    def corners(self) -> NDArray:
        return self._base.corners

    # ---------------------------------- BASE METHODS ---------------------------------

    def as_grayscale(self) -> Self:
        self._base.as_grayscale()
        return self

    def as_colorscale(self) -> Self:
        self._base.as_colorscale()
        return self

    def as_filled(self, fill_value: int | np.ndarray = 255) -> Self:
        self._base.as_filled(fill_value=fill_value)
        return self

    def as_white(self) -> Self:
        self._base.as_white()
        return self

    def as_black(self) -> Self:
        self._base.as_black()
        return self

    def as_pil(self) -> ImagePIL.Image:
        return self._base.as_pil()

    def rev(self) -> Self:
        self._base.rev()
        return self

    def dist_pct(self, pct: float) -> float:
        return self._base.dist_pct(pct=pct)

    def is_equal_shape(self, other: Image, consider_channel: bool = True) -> bool:
        return self._base.is_equal_shape(
            other=other._base, consider_channel=consider_channel
        )

    # ---------------------------------- COPY METHOD ----------------------------------

    def copy(self) -> Image:
        """Copy of the image.

        For NumPy arrays containing basic data types (e.g., int, float, bool),
        using copy.deepcopy() is generally unnecessary.
        The numpy.copy() method achieves the same result more efficiently.
        numpy.copy() creates a new array in memory with a separate copy of the data,
        ensuring that modifications to the copy do not affect the original array.

        Returns:
            Image: image copy
        """
        return Image(image=self.asarray.copy())

    # -------------------------------- WRITE METHODS ----------------------------------

    def save(self, save_filepath: str) -> None:
        self.writer.save(save_filepath=save_filepath)

    def show(
        self,
        title: Optional[str] = None,
        figsize: tuple[float, float] = (8.0, 6.0),
        color_conversion: int = cv2.COLOR_BGR2RGB,
        save_filepath: Optional[str] = None,
    ) -> None:
        self.writer.show(
            title=title,
            figsize=figsize,
            color_conversion=color_conversion,
            save_filepath=save_filepath,
        )

    # -------------------------------- DRAWER METHODS ---------------------------------

    def draw_circles(
        self, circles: list[geo.Circle], render: CirclesRender = CirclesRender()
    ) -> Self:
        self.drawer.draw_circles(circles=circles, render=render)
        return self

    def draw_points(
        self,
        points: np.ndarray | list[geo.Point],
        render: PointsRender = PointsRender(),
    ) -> Self:
        self.drawer.draw_points(points=points, render=render)
        return self

    def draw_segments(
        self,
        segments: np.ndarray | list[geo.Segment],
        render: SegmentsRender = SegmentsRender(),
    ) -> Self:
        self.drawer.draw_segments(segments=segments, render=render)
        return self

    def draw_polygons(
        self,
        polygons: list[geo.Polygon],
        render: PolygonsRender = PolygonsRender(),
    ) -> Self:
        self.drawer.draw_polygons(polygons=polygons, render=render)
        return self

    def draw_splines(
        self,
        splines: list[geo.LinearSpline],
        render: LinearSplinesRender = LinearSplinesRender(),
    ) -> Self:
        self.drawer.draw_splines(splines=splines, render=render)
        return self

    def draw_ocr_outputs(
        self,
        ocr_outputs: list[OcrSingleOutput],
        render: OcrSingleOutputRender = OcrSingleOutputRender(),
    ) -> Self:
        self.drawer.draw_ocr_outputs(ocr_outputs=ocr_outputs, render=render)
        return self

    # --------------------------------- BINARIZER -------------------------------------

    def threshold_simple(self, thresh: int) -> Self:
        self.transformer.binarizer.threshold_simple(thresh=thresh)
        return self

    def threshold_adaptative(self) -> Self:
        self.transformer.binarizer.threshold_adaptative()
        return self

    def threshold_otsu(self) -> Self:
        self.transformer.binarizer.threshold_otsu()
        return self

    def threshold_niblack(self, window_size: int = 15, k: float = 0.2) -> Self:
        self.transformer.binarizer.threshold_niblack(window_size=window_size, k=k)
        return self

    def threshold_sauvola(
        self, window_size: int = 15, k: float = 0.2, r: float = 128.0
    ) -> Self:
        self.transformer.binarizer.threshold_sauvola(window_size=window_size, k=k, r=r)
        return self

    def binary(self, method: BinarizationMethods = "sauvola") -> NDArray:
        return self.transformer.binarizer.binary(method=method)

    def binaryrev(self, method: BinarizationMethods = "sauvola") -> NDArray:
        return self.transformer.binarizer.binaryrev(method=method)

    # ---------------------------------- CROPPER --------------------------------------
    # the copy arguments is special in the crop methods.
    # this is important for performance reasons
    # if you want to crop a small part of an image and conserve the original
    # without doing image.copy().crop() which would copy the entire original image!
    # this would be much more expensive if the image is large

    def crop(
        self, x0: int, y0: int, x1: int, y1: int, copy: bool = False, **kwargs
    ) -> Image | Self:
        out = self.transformer.cropper.crop(
            x0=x0, y0=y0, x1=x1, y1=y1, copy=copy, **kwargs
        )
        return out if copy else self

    def crop_from_topleft(
        self, topleft: NDArray, width: int, height: int, copy: bool = False, **kwargs
    ) -> Image | Self:
        out = self.transformer.cropper.crop_from_topleft(
            topleft=topleft, width=width, height=height, copy=copy, **kwargs
        )
        return out if copy else self

    def crop_from_center(
        self, center: NDArray, width: int, height: int, copy: bool = False, **kwargs
    ) -> Image | Self:
        out = self.transformer.cropper.crop_from_center(
            center=center, width=width, height=height, copy=copy, **kwargs
        )
        return out if copy else self

    def crop_from_axis_aligned_bbox(
        self, bbox: geo.Rectangle, copy: bool = False, **kwargs
    ) -> Image | Self:
        out = self.transformer.cropper.crop_from_axis_aligned_bbox(
            bbox=bbox, copy=copy, **kwargs
        )
        return out if copy else self

    def crop_from_polygon(
        self, polygon: geo.Polygon, copy: bool = False, **kwargs
    ) -> Image | Self:
        out = self.transformer.cropper.crop_from_polygon(
            polygon=polygon, copy=copy, **kwargs
        )
        return out if copy else self

    def crop_from_linear_spline(
        self, spline: geo.LinearSpline, copy: bool = False, **kwargs
    ) -> Image | Self:
        out = self.transformer.cropper.crop_from_linear_spline(
            spline=spline, copy=copy, **kwargs
        )
        return out if copy else self

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
        im = self.copy()  # the copy before makes this method slow

        # center the image based on the middle of the line
        geo_segment = geo.Segment(segment)
        im, translation_vector = im.center_to_segment(segment=segment)

        if width_crop_rect == -1:
            # default the width for crop to be a bit more than line length
            width_crop_rect = int(geo_segment.length)
        width_crop_rect += added_width
        assert width_crop_rect > 0 and height_crop_rect > 0

        # rotate the image so that the line is horizontal
        angle = geo_segment.slope_angle(is_y_axis_down=True)
        im = im.rotate(angle=angle)

        # cropping
        im_crop = im.crop_from_center(
            center=im._base.center,
            width=width_crop_rect,
            height=height_crop_rect,
        )

        crop_translation_vector = self._base.center - im_crop._base.center
        return im_crop, translation_vector, angle, crop_translation_vector

    def crop_around_segment_horizontal_faster(
        self,
        segment: np.ndarray,
        dim_crop_rect: tuple[int, int] = (-1, 100),
        added_width: int = 75,
        pad_value: int = 0,
    ) -> Image:
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
        angle = geo_segment.slope_angle(is_y_axis_down=True)

        if width_crop_rect == -1:
            # default the width for crop to be a bit more than line length
            width_crop_rect = int(geo_segment.length)
        width_crop_rect += added_width
        assert width_crop_rect > 0 and height_crop_rect > 0

        x_extra = abs(added_width / 2 * np.cos(angle))
        y_extra = abs(added_width / 2 * np.sin(angle))

        # add extra width for crop in case segment is ~vertical
        x_extra += int(width_crop_rect / 2) + 1
        y_extra += int(height_crop_rect / 2) + 1

        im: Image = self.crop(
            x0=geo_segment.xmin - x_extra,
            y0=geo_segment.ymin - y_extra,
            x1=geo_segment.xmax + x_extra,
            y1=geo_segment.ymax + y_extra,
            pad=True,
            clip=False,
            copy=True,  # copy the image after cropping for very fast performance
            pad_value=pad_value,
        )

        # rotate the image so that the line is horizontal
        im.rotate(angle=angle)

        # cropping around segment center
        im.crop_from_center(
            center=im._base.center,
            width=width_crop_rect,
            height=height_crop_rect,
        )

        return im

    def crop_next_to_rectangle(
        self,
        rect: geo.Rectangle,
        rect_topleft_ix: int,
        crop_dim: tuple[int, int] = (-1, -1),
        crop_shift: tuple[int, int] = (0, 0),
    ) -> Self:
        """Crop image in the referential of the rectangle.

        Args:
            rect (geo.Rectangle): rectangle for reference to crop.
            rect_topleft_ix (int): top-left vertice index of the rectangle
            crop_dim (tuple[int, int], optional): (width, height) crop dimension.
                Defaults to (-1, -1).
            crop_shift (tuple[int, int], optional): The shift is (x, y).
                The crop_shift argument is applied from the rectangle center based on
                the axis referential of the rectangle.
                This means that the shift in the Y direction
                is based on the normalized vector (bottom-left, top-left)
                The shift in the X direction is based on the normalized vector
                (top-left, top-right). Defaults to (0, 0) meaning no shift.

        Returns:
            Self: new image cropped
        """
        # shift down and up vector calculated based on the top-left vertice
        rect_shift_up = rect.get_vector_up_from_topleft(topleft_index=rect_topleft_ix)
        rect_shift_left = rect.get_vector_left_from_topleft(
            topleft_index=rect_topleft_ix
        )

        # crop dimension
        rect_heigth = rect.get_height_from_topleft(topleft_index=rect_topleft_ix)
        crop_width = rect_heigth if crop_dim[0] == -1 else crop_dim[0]
        crop_height = rect_heigth if crop_dim[1] == -1 else crop_dim[1]
        crop_width, crop_height = int(crop_width), int(crop_height)
        assert crop_width > 0 and crop_height > 0

        # compute the crop center
        crop_center = rect.centroid
        crop_center += crop_shift[0] * rect_shift_left.normalized  # shift left
        crop_center += crop_shift[1] * rect_shift_up.normalized  # shift up

        # get the crop segment
        crop_segment = geo.Segment(
            [
                crop_center - crop_width / 2 * rect_shift_left.normalized,
                crop_center + crop_width / 2 * rect_shift_left.normalized,
            ]
        )

        return self.crop_around_segment_horizontal_faster(
            segment=crop_segment.asarray,
            dim_crop_rect=(crop_width, crop_height),
            added_width=0,
        )

    # ------------------------------- GEOMETRY METHODS --------------------------------

    def shift(
        self, shift: NDArray, border_fill_value: Sequence[float] = (0.0,)
    ) -> Self:
        self.transformer.geometrizer.shift(
            shift=shift, border_fill_value=border_fill_value
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
        self.transformer.geometrizer.rotate(
            angle=angle,
            is_degree=is_degree,
            is_clockwise=is_clockwise,
            reshape=reshape,
            border_fill_value=border_fill_value,
            fast=fast,
        )
        return self

    def center_to_point(self, point: NDArray) -> tuple[Self, NDArray]:
        shift_vector = self.transformer.geometrizer.center_to_point(point=point)
        return self, shift_vector

    def center_to_segment(self, segment: NDArray) -> tuple[Self, NDArray]:
        shift_vector = self.transformer.geometrizer.center_to_segment(segment=segment)
        return self, shift_vector

    def restrict_rect_in_frame(self, rectangle: geo.Rectangle) -> geo.Rectangle:
        return self.transformer.geometrizer.restrict_rect_in_frame(rectangle=rectangle)

    # ----------------------------- MORPHOLOGICAL METHODS -----------------------------

    def resize_fixed(
        self,
        dim: tuple[int, int],
        interpolation: int = cv2.INTER_AREA,
        copy: bool = False,
    ) -> Image | Self:
        out = self.transformer.morphologyzer.resize_fixed(
            dim=dim, interpolation=interpolation, copy=copy
        )
        return out if copy else self

    def resize(
        self, factor: float, interpolation: int = cv2.INTER_AREA, copy: bool = False
    ) -> Image | Self:
        out = self.transformer.morphologyzer.resize(
            factor=factor, interpolation=interpolation, copy=copy
        )
        return out if copy else self

    def blur(
        self,
        kernel: tuple = (5, 5),
        iterations: int = 1,
        method: BlurMethods = "average",
        sigmax: float = 0,
    ) -> Self:
        self.transformer.morphologyzer.blur(
            kernel=kernel, iterations=iterations, method=method, sigmax=sigmax
        )
        return self

    def dilate(
        self,
        kernel: tuple = (5, 5),
        iterations: int = 1,
        dilate_black_pixels: bool = True,
    ) -> Self:
        self.transformer.morphologyzer.dilate(
            kernel=kernel,
            iterations=iterations,
            dilate_black_pixels=dilate_black_pixels,
        )
        return self

    def erode(
        self,
        kernel: tuple = (5, 5),
        iterations: int = 1,
        erode_black_pixels: bool = True,
    ) -> Self:
        self.transformer.morphologyzer.erode(
            kernel=kernel, iterations=iterations, erode_black_pixels=erode_black_pixels
        )
        return self

    # -------------------------- ASSEMBLED COMPOSED METHODS ---------------------------
    # methods that use multiple components
    # ---------------------------------------------------------------------------------

    def iou(
        self, other: Image, binarization_method: BinarizationMethods = "sauvola"
    ) -> float:
        """Compute the intersection over union score

        Args:
            other (Image): another image
            binarization_method (str, optional): binarization method to turn images
                into 0 and 1 images. The black pixels will be 1 and the white pixels
                will be 0. This is used to compute the score.
                Defaults to "sauvola".

        Returns:
            float: a score from 0 to 1. The greater the score the greater the other
                image is equal to the original image
        """
        assert self.is_equal_shape(other)
        mask0 = self.binaryrev(method=binarization_method)
        mask1 = other.binaryrev(method=binarization_method)
        return np.sum(mask0 * mask1) / np.count_nonzero(mask0 + mask1)

    def score_contains_v2(
        self, other: Image, binarization_method: BinarizationMethods = "sauvola"
    ) -> float:
        """Score contains version 2 which is more efficient and faster.

        Args:
            other (Image): other Image object
            binarization_method (str, optional): binarization method to turn images
                into 0 and 1 images. The black pixels will be 1 and the white pixels
                will be 0. This is used to compute the score.
                Defaults to "sauvola".

        Returns:
            float: a score from 0 to 1. The greater the score the greater the other
                image is contained within the original image
        """
        assert self.is_equal_shape(other, consider_channel=False)

        cur_binaryrev = self.binaryrev(method=binarization_method)
        other_binaryrev = other.binaryrev(method=binarization_method)

        other_pixels = cur_binaryrev[other_binaryrev == 1]

        coverage = np.sum(other_pixels) / np.sum(other_binaryrev)
        return coverage

    def score_contains(
        self, other: Image, binarization_method: BinarizationMethods = "sauvola"
    ) -> float:
        """How much the other image is contained in the original image.

        Args:
            other (Image): other Image object
            binarization_method (str, optional): binarization method to turn images
                into 0 and 1 images. The black pixels will be 1 and the white pixels
                will be 0. This is used to compute the score.
                Defaults to "sauvola".

        Returns:
            float: a score from 0 to 1. The greater the score the greater the other
                image is contained within the original image
        """
        assert self.is_equal_shape(other, consider_channel=False)
        other_binaryrev = other.binaryrev(method=binarization_method)
        return np.sum(
            self.binaryrev(method=binarization_method) * other_binaryrev
        ) / np.sum(other_binaryrev)

    def score_contains_segments_v2(
        self,
        segments: list[geo.Segment],
        dilate_kernel: tuple = (5, 5),
        dilate_iterations: int = 0,
        binarization_method: BinarizationMethods = "sauvola",
    ) -> list[float]:
        """Compute the contains score in [0, 1] for each individual segment.
        This method can be better than :func:`~Image.score_contains_polygons()` in some
        cases.
        It provides a score for each single segments. This way it is better to
        identify which segments specifically are contained in the image or not.

        Args:
            segments (np.ndarray | list[geo.Segment]): a list of segments
            dilate_kernel (tuple, optional): dilate kernel param. Defaults to (5, 5).
            dilate_iterations (int, optional): dilate iterations param. Defaults to 0.
            binarization_method (str, optional): binarization method. Here
                we can afford the sauvola method since we crop the image first
                and the binarization occurs on a small image.
                Defaults to "sauvola".

        Returns:
            np.ndarray: list of score for each individual segment in the same order
                as the list of segments
        """
        # pylint: disable=too-many-arguments, too-many-positional-arguments
        score_segments: list[float] = []

        im = self.copy().dilate(kernel=dilate_kernel, iterations=dilate_iterations)
        im_white = im.copy().as_white()

        for segment in segments:

            # create all-white image of same size as original with the segment drawn
            other = im_white.draw_segments(
                segments=[segment],
                render=SegmentsRender(thickness=1, default_color=(0, 0, 0)),
            ).as_grayscale()

            score = im.score_contains_v2(
                other=other, binarization_method=binarization_method
            )

            score_segments.append(score)

        return score_segments

    def score_contains_segments(
        self,
        segments: list[geo.Segment],
        dilate_kernel: tuple = (5, 5),
        dilate_iterations: int = 0,
        binarization_method: BinarizationMethods = "sauvola",
        resize_factor: float = 1.0,
    ) -> list[float]:
        """Compute the contains score in [0, 1] for each individual segment.
        This method can be better than :func:`~Image.score_contains_polygons()` in some
        cases.
        It provides a score for each single segments. This way it is better to
        identify which segments specifically are contained in the image or not.

        Args:
            segments (np.ndarray | list[geo.Segment]): a list of segments
            dilate_kernel (tuple, optional): dilate kernel param. Defaults to (5, 5).
            dilate_iterations (int, optional): dilate iterations param. Defaults to 0.
            binarization_method (str, optional): binarization method. Here
                we can afford the sauvola method since we crop the image first
                and the binarization occurs on a small image.
                Defaults to "sauvola".
            resize_factor (float, optional): resize factor that can be adjusted to
                provide extra speed. A lower value will be faster but less accurate.
                Typically 0.5 works well but less can have a negative impact on accuracy
                Defaults to 1.0 which implies no resize.

        Returns:
            np.ndarray: list of score for each individual segment in the same order
                as the list of segments
        """
        # pylint: disable=too-many-arguments, too-many-positional-arguments
        added_width = 10
        height_crop = 30
        mid_height_crop = int(height_crop / 2)
        score_segments: list[float] = []

        for segment in segments:

            im = self.crop_around_segment_horizontal_faster(
                segment=segment.asarray,
                dim_crop_rect=(-1, height_crop),
                added_width=added_width,
                pad_value=255,
            )

            im.dilate(kernel=dilate_kernel, iterations=dilate_iterations)

            im.resize(factor=resize_factor)

            # re-compute the segment in the crop referential
            segment_crop = geo.Segment(
                np.array(
                    [
                        [added_width, mid_height_crop],
                        [segment.length + added_width, mid_height_crop],
                    ]
                )
                * resize_factor
            )

            # create all-white image of same size as original with the segment drawn
            other = (
                im.copy()
                .as_white()
                .draw_segments(
                    segments=[segment_crop],
                    render=SegmentsRender(thickness=1, default_color=(0, 0, 0)),
                )
                .as_grayscale()
            )

            score = im.score_contains_v2(
                other=other, binarization_method=binarization_method
            )

            score_segments.append(score)

        return score_segments

    def score_contains_polygons(
        self,
        polygons: list[geo.Polygon],
        dilate_kernel: tuple = (5, 5),
        dilate_iterations: int = 0,
        binarization_method: BinarizationMethods = "sauvola",
        resize_factor: float = 1.0,
    ) -> list[float]:
        """Compute the contains score in [0, 1] for each individual polygon.

        Beware: this method is different from the score_contains method because in
        this case you can emphasize the base image by dilating its content.

        Everything that is a 1 in the rmask will be dilated to give more chance for the
        contour to be contained within the image in the calculation. This way you
        can control the sensitivity of the score.

        Args:
            polygon (Polygon): Polygon object
            dilate_kernel (tuple, optional): dilate kernel param. Defaults to (5, 5).
            dilate_iterations (int, optional): dilate iterations param. Defaults to 0.
            binarization_method (str, optional): binarization method. Here
                we can afford the sauvola method since we crop the image first
                and the binarization occurs on a small image.
                Defaults to "sauvola".
            resize_factor (float, optional): resize factor that can be adjusted to
                provide extra speed. A lower value will be faster but less accurate.
                Typically 0.5 works well but less can have a negative impact on accuracy
                Defaults to 1.0 which implies no resize.

        Returns:
            float: a score from 0 to 1. The greater the score the greater the contour
                 is contained within the original image
        """
        # pylint: disable=too-many-arguments, too-many-positional-arguments
        extra_border_size = 10
        scores: list[float] = []
        for polygon in polygons:

            im = self.crop_from_polygon(
                polygon=polygon,
                copy=True,
                pad=True,
                clip=False,
                extra_border_size=extra_border_size,
                pad_value=255,
            )

            im.dilate(kernel=dilate_kernel, iterations=dilate_iterations)

            im.resize(factor=resize_factor)

            # re-compute the polygon in the crop referential
            polygon_crop = geo.Polygon(
                (polygon.crop_coordinates + extra_border_size) * resize_factor
            )

            # create all-white image of same size as original with the geometry entity
            other = (
                im.copy()
                .as_white()
                .draw_polygons(
                    polygons=[polygon_crop],
                    render=PolygonsRender(thickness=1, default_color=(0, 0, 0)),
                )
                .as_grayscale()
            )

            cur_score = im.score_contains_v2(
                other=other, binarization_method=binarization_method
            )

            scores.append(cur_score)

        return scores

    def score_contains_linear_splines(
        self,
        splines: list[geo.LinearSpline],
        dilate_kernel: tuple = (5, 5),
        dilate_iterations: int = 0,
        binarization_method: BinarizationMethods = "sauvola",
        resize_factor: float = 1.0,
    ) -> list[float]:
        """Compute the contains score in [0, 1]for each individual LinearSpline.
        It provides a score for each single linear spline.

        Args:
            segments (np.ndarray | list[geo.Segment]): a list of segments
            dilate_kernel (tuple, optional): dilate kernel param. Defaults to (5, 5).
            dilate_iterations (int, optional): dilate iterations param. Defaults to 0.
            binarization_method (str, optional): binarization method. Here
                we can afford the sauvola method since we crop the image first
                and the binarization occurs on a small image.
                Defaults to "sauvola".
            resize_factor (float, optional): resize factor that can be adjusted to
                provide extra speed. A lower value will be faster but less accurate.
                Typically 0.5 works well but less can have a negative impact on accuracy
                Defaults to 1.0 which implies no resize.
        """
        # pylint: disable=too-many-arguments, too-many-positional-arguments
        extra_border_size = 10
        scores: list[float] = []
        for spline in splines:
            im = self.crop_from_linear_spline(
                spline=spline,
                copy=True,
                pad=True,
                clip=False,
                extra_border_size=extra_border_size,
                pad_value=255,
            )

            im.dilate(kernel=dilate_kernel, iterations=dilate_iterations)

            im.resize(factor=resize_factor)

            spline_crop = geo.LinearSpline(
                (spline.crop_coordinates + extra_border_size) * resize_factor
            )

            # create all-white image of same size as original with the geometry entity
            other = (
                im.copy()
                .as_white()
                .draw_splines(
                    splines=[spline_crop],
                    render=LinearSplinesRender(thickness=1, default_color=(0, 0, 0)),
                )
                .as_grayscale()
            )

            cur_score = im.score_contains_v2(
                other=other, binarization_method=binarization_method
            )

            scores.append(cur_score)

        return scores

    def score_contains_linear_entities(
        self,
        entities: list[LinearEntity],
        dilate_kernel: tuple = (5, 5),
        dilate_iterations: int = 0,
        binarization_method: BinarizationMethods = "sauvola",
        resize_factor: float = 1.0,
    ) -> list[float]:
        """Compute the contains score in [0, 1] for each individual linear entity
        (either LinearSpline or Segment).

        Args:
            entities (list[geo.LinearEntity]): a list of linear entities
                (splines or segments)
            dilate_kernel (tuple, optional): dilate kernel param. Defaults to (5, 5).
            dilate_iterations (int, optional): dilate iterations param. Defaults to 0.
            binarization_method (BinarizationMethods, optional): binarization method.
                Defaults to "sauvola".
            resize_factor (float, optional): resize factor for speed/accuracy tradeoff.
                Defaults to 1.0.

        Returns:
            list[float]: list of scores for each individual entity
        """
        # pylint: disable=too-many-arguments, too-many-positional-arguments
        scores = []
        for entity in entities:
            if isinstance(entity, geo.LinearSpline):
                score = self.score_contains_linear_splines(
                    splines=[entity],
                    dilate_kernel=dilate_kernel,
                    dilate_iterations=dilate_iterations,
                    binarization_method=binarization_method,
                    resize_factor=resize_factor,
                )[0]
            elif isinstance(entity, geo.Segment):
                score = self.score_contains_segments(
                    segments=[entity],
                    dilate_kernel=dilate_kernel,
                    dilate_iterations=dilate_iterations,
                    binarization_method=binarization_method,
                    resize_factor=resize_factor,
                )[0]
            else:
                raise TypeError(
                    f"Unsupported entity type: {type(entity)}. "
                    "Expected LinearSpline or Segment."
                )
            scores.append(score)
        return scores

    def score_distance_from_center(
        self, point: np.ndarray, method: ScoreDistanceFromCenterMethods = "linear"
    ) -> float:
        """Compute a score to evaluate how far a point is from the
        image center point.

        A score close to 0 means that the point and the image center are far away.
        A score close to 1 means that the point and the image center are close.

        It is particularly useful when calling it where the point argument is a
        contour centroid. Then, a score equal to 1 means that the contour and image
        centers coincide.

        This method can be used to compute a score for a contour centroid:
        - A small score should be taken into account and informs us that the contour
        found is probably wrong.
        - On the contrary, a high score does not ensure a high quality contour.

        Args:
            point (np.ndarray): 2D point
            method (str): the method to be used to compute the score. Defaults to
                "linear".

        Returns:
            float: a score from 0 to 1.
        """

        def gaussian_2d(
            x: float,
            y: float,
            x0: float = 0.0,
            y0: float = 0.0,
            amplitude: float = 1.0,
            sigmax: float = 1.0,
            sigmay: float = 1.0,
        ) -> float:
            # pylint: disable=too-many-positional-arguments,too-many-arguments
            return amplitude * np.exp(
                -((x - x0) ** 2 / (2 * sigmax**2) + (y - y0) ** 2 / (2 * sigmay**2))
            )

        def cone_positive_2d(
            x: float,
            y: float,
            x0: float = 0.0,
            y0: float = 0.0,
            amplitude: float = 1.0,
            radius: float = 1.0,
        ) -> float:
            # pylint: disable=too-many-positional-arguments,too-many-arguments
            r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
            if r >= radius:
                return 0
            return amplitude * (1 - r / radius)

        if method == "linear":
            return cone_positive_2d(
                x=point[0],
                y=point[1],
                x0=self.center[0],
                y0=self.center[1],
                radius=self.norm_side_length / 2,
            )
        if method == "gaussian":
            return gaussian_2d(
                x=point[0],
                y=point[1],
                x0=self.center[0],
                y0=self.center[1],
                sigmax=self.dist_pct(0.1),
                sigmay=self.dist_pct(0.1),
            )

        raise ValueError(f"Unknown method {method}")
