"""
File utils for image manipulation
"""

from __future__ import annotations

from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage

import src.geometry as geo
from src.utils.dataclasses.OcrSingleOutput import OcrSingleOutput


class Image:
    def __init__(self, image: np.ndarray | Image) -> None:
        if isinstance(image, Image):
            image = image.asarray
        self.asarray: np.ndarray = image.copy()

    @property
    def is_gray(self):
        return len(self.asarray.shape) == 2

    @property
    def height(self):
        return self.asarray.shape[0]

    @property
    def width(self):
        return self.asarray.shape[1]

    @property
    def area(self):
        return self.width * self.height

    @property
    def center(self):
        # return as type int because the center needs to represent XY coords of a pixel
        return (np.array([self.width, self.height]) / 2).astype(int)

    @property
    def norm_side_length(self):
        return np.sqrt(self.area)

    def margin_distance_error(self, pct: float = 0.01):
        return self.norm_side_length * pct

    def copy(self) -> Image:
        return Image(self.asarray.copy())

    # ------------------------------- DISPLAY METHODS ----------------------------------

    def show(
        self,
        title: Optional[str] = None,
        figsize: tuple[float, float] = (8.0, 6.0),
        color_conversion: int = cv2.COLOR_BGR2RGB,
        save_filepath: Optional[str] = None,
    ) -> None:
        """Display the image

        Args:
            title (str, optional): optional title for the image. Defaults to None.
            conversion (int, optional): color conversion. Defaults to cv2.COLOR_BGR2RGB.
        """
        # Converts from one colour space to the other. this is needed as RGB
        # is not the default colour space for OpenCV
        if color_conversion is not None:
            im = cv2.cvtColor(self.asarray, color_conversion)

        plt.figure(figsize=figsize)

        # Show the image
        plt.imshow(im)

        # remove the axis / ticks for a clean looking image
        plt.xticks([])
        plt.yticks([])

        # if a title is provided, show it
        if title is not None:
            plt.title(title)

        if save_filepath is not None:
            plt.savefig(save_filepath)

        plt.show()

    # --------------------------------- DRAWING METHODS --------------------------------

    def add_points(
        self,
        points: np.ndarray,
        colors_points: Optional[list] = None,
        default_color=(0, 0, 255),
        radius: int = 3,
        thickness: int = 3,
    ) -> Image:
        """Add points on an image

        Args:
            points (np.ndarray): points
            colors_points (np.ndarray, optional): color for each color.
                Defaults to None.
            default_color (tuple, optional): default color for points.
                Defaults to (0, 0, 255).
            radius (int, optional): radius of the points. Defaults to 3.
            thickness (int, optional): thickness of the point. Defaults to 3.

        Returns:
            Image: new image
        """
        im = self.asarray.copy()

        # needs to be in type int to define precise pixels
        points = np.asarray(points).astype(int)

        # define color of the line to be default color in case not define in args
        if colors_points is None:
            colors_points = [default_color for i in range(len(points))]

        if len(points) != len(colors_points):
            raise RuntimeError(
                f"The number of points ({len(points)}) do not coincide with the number "
                f"of colors ({len(colors_points)})"
            )

        # draw points in image
        if Image(im).is_gray:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        for pt, color in zip(points, colors_points):
            cv2.circle(
                img=im, center=pt, radius=radius, color=color, thickness=thickness
            )

        return Image(im)

    def add_segments(
        self,
        segments: np.ndarray,
        as_vectors: bool = False,
        colors_segments: Optional[list] = None,
        default_color=(0, 0, 255),
        thickness: int = 3,
        line_type: int = cv2.LINE_AA,
        tip_length: int = 20,
    ) -> Image:
        """Add segments on an image. It can be arrowed segments (vectors) too.

        Args:
            segments (np.ndarray): list of segments
            as_vectors (bool, optional): whether to display segments as vectors with
                a direction. The second coordinate define the tip of the arrow.
                Defaults to False.
            colors_segments (np.ndarray, optional): color of each segment.
                Defaults to None.
            default_color (tuple, optional): tuple in BGR space.
                Defaults to (0, 0, 255).
            thickness (int, optional): display thickness of the segments.
                Defaults to 3.
            line_type (int, optional): display line type. Defaults to cv2.LINE_AA.
            tip_length (int, optional): size of the arrow tip

        Returns:
            (Image): a new image that contains the segments drawn
        """
        im = self.asarray.copy()

        # needs to be in type int to define precise pixels
        segments = np.asarray(segments).astype(int)

        # default color in case not define in args
        if colors_segments is None:
            colors_segments = [default_color for i in range(len(segments))]

        if len(segments) != len(colors_segments):
            raise RuntimeError(
                f"The number of segments ({len(segments)}) do not coincide with the "
                f"number of colors ({len(colors_segments)})"
            )

        # draw segments in image
        if Image(im).is_gray:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        for seg, color in zip(segments, colors_segments):
            if as_vectors:
                cv2.arrowedLine(
                    im,
                    seg[0],
                    seg[1],
                    color,
                    thickness,
                    line_type,
                    tipLength=tip_length / geo.Segment(seg).length,
                )
            else:
                cv2.line(im, seg[0], seg[1], color, thickness, line_type)

        return Image(im)

    def add_contours(
        self,
        contours: list[geo.Contour],
        as_vectors: bool = False,
        colors_segments: Optional[list] = None,
        default_color=(0, 0, 255),
        thickness: int = 3,
        line_type: int = cv2.LINE_AA,
        tip_length: int = 20,
    ):
        im = self.copy()

        for cnt in contours:
            im.add_segments(
                segments=cnt.lines,
                as_vectors=as_vectors,
                colors_segments=colors_segments,
                default_color=default_color,
                thickness=thickness,
                line_type=line_type,
                tip_length=tip_length,
            )

        return Image(im)

    def add_ocr_outputs(
        self,
        ocr_outputs: list[OcrSingleOutput],
        default_bbox_color: tuple = (0, 0, 255),
        thickness: int = 2,
    ) -> Image:
        """Return a new image with the bounding boxes displayed from a list of OCR
        single output. It allows you to show bounding boxes that can have an angle,
        not necessarily vertical or horizontal.

        Args:
            ocr_outputs (list[OcrSingleOutput]): list of OcrSingleOutput dataclass.
            default_bbox_color (tuple, optional): color for the bounding boxes.
                Defaults to (125, 125, 230).

        Returns:
            (Image): a new image with the bounding boxes displayed
        """
        im = self.asarray.copy()
        if Image(im).is_gray:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        for ocrso in ocr_outputs:
            cnt = [ocrso.bbox.asarray.reshape((-1, 1, 2)).astype(np.int32)]
            im = cv2.drawContours(
                im,
                contours=cnt,
                contourIdx=-1,
                thickness=thickness,
                color=default_bbox_color,
            )
        return Image(im)

    # ------------------------------ TRANSFORMATION METHODS ----------------------------

    def shift(self, shift: np.ndarray, mode: str = "constant") -> Image:
        """Shift the image doing a translation operation

        Args:
            shift (np.ndarray): Vector for translation
            mode (str, optional): Defaults to "contants"

        Returns:
            Image: new image translated
        """
        im = scipy.ndimage.shift(
            input=self.asarray, shift=geo.Vector(shift).cv2_space_coords, mode=mode
        )
        return Image(im)

    def rotate(self, angle: float, is_degree: bool = True, reshape: bool = True):
        """Rotate the image

        Args:
            angle (float): angle to rotate the image
            is_degree (bool, optional): whether the angle is in degree or not.
                Defaults to True.
            reshape (bool, optional): scipy reshape option. Defaults to True.

        Returns:
            (Image): new Image rotated
        """
        if not is_degree:
            angle = np.rad2deg(angle)
        im = scipy.ndimage.rotate(input=self.asarray, angle=angle, reshape=reshape)
        return Image(im)

    def center_image_to_point(
        self,
        point: np.ndarray,
        mode: str = "constant",
        return_shift_vector: bool = False,
    ) -> Image | tuple[Image, np.ndarray]:
        """Shift the image so that the input point ends up in the middle of the
        new image

        Args:
            point (np.ndarray): point as (2,) shape numpy array
            mode (str, optional): scipy shift interpolation mode.
                Defaults to "constant".

        Returns:
            (tuple[Image, np.ndarray]): Image, translation Vector
        """
        shift_vector = self.center - point
        im = self.shift(shift=shift_vector, mode=mode)
        if return_shift_vector:
            return im, shift_vector
        else:
            return im

    def center_image_to_segment(
        self,
        segment: np.ndarray,
        mode: str = "constant",
        return_shift_vector: bool = False,
    ) -> Image | tuple[Image, np.ndarray]:
        """Shift the image so that the line middle point ends up in the middle
        of the new image

        Args:
            segment (Segment): segment class object
            mode (str, optional): scipy mode for the translation.
                Defaults to "constant".

        Returns:
            (tuple[Image, np.ndarray]): Image, vector_shift
        """
        return self.center_image_to_point(
            point=geo.Segment(segment).centroid,
            mode=mode,
            return_shift_vector=return_shift_vector,
        )

    def resize_fixed(
        self, dim: tuple[int, int], interpolation: int = cv2.INTER_AREA
    ) -> Image:
        im = self.asarray.copy()
        im = cv2.resize(src=im, dsize=dim, interpolation=interpolation)
        return Image(im)

    def resize(
        self, scale_percent: float, interpolation: int = cv2.INTER_AREA
    ) -> Image:
        """Resize the image to a new size

        Args:
            scale_percent (float): scale to resize the image. A value 100 does not
                change the image. 200 double the image size.

        Returns:
            (Image): new resized image
        """
        if scale_percent == 100:
            return self

        width = int(self.width * scale_percent / 100)
        height = int(self.height * scale_percent / 100)
        dim = (width, height)
        return self.resize_fixed(dim=dim, interpolation=interpolation)

    def crop_image_horizontal(self, x0: int, y0: int, x1: int, y1: int) -> Image:
        """Crop an image

        Args:
            x0 (int): x coordinate of the first point
            y0 (int): y coordinate of the first point
            x1 (int): x coordinate of the second point
            y1 (int): y coordinate of the second point

        Returns:
            Image: new image cropped
        """
        im = self.asarray.copy()
        im = im[x0:x1, y0:y1]
        return Image(im)

    def crop_around_segment_horizontal(
        self,
        segment: np.ndarray,
        heigth_crop_rect: int = 100,
        width_crop_rect: Optional[int] = None,
        default_extra_width: int = 75,
        show_translation: bool = False,
        return_transfo_bricks: bool = False,
    ) -> Image | tuple[Image, np.ndarray, float, np.ndarray]:
        """Crop around a specific segment in the image. This is done in three
        specific steps:
        1) shift image so that the middle of the segment is in the middle of the image
        2) rotate image by the angle of segment so that the segment becomes horizontal
        3) crop the image

        Args:
            segment (np.ndarray): Segment class object
            heigth_crop_rect (int, optional): height. Defaults to 100.
            width_crop_rect (int, optional): width. Defaults to None.
            default_extra_width (int, optional): additional width for cropping.
                Defaults to 75.
            show_translation (bool, optional): whether to show translation or not.
                Defaults to False.

        Returns:
            tuple[Image, np.ndarray, float]: _description_
        """
        im = self.copy()

        # center the image based on the middle of the line
        geo_segment = geo.Segment(segment)
        im, translation_vector = im.center_image_to_segment(
            segment=segment, return_shift_vector=True
        )

        if show_translation:
            self.add_segments(segments=translation_vector, as_vectors=True).show()

        # rotate the image so that the line is horizontal
        angle = geo_segment.slope_angle(degree=True)
        im = im.rotate(angle=angle)

        if width_crop_rect is None:
            # default the width for crop to be a bit more than line length
            width_crop_rect = int(geo_segment.length + default_extra_width)

        # calculate the coordinates of the rectangle for cropping
        x0, y0 = (
            int(im.center[1] - heigth_crop_rect / 2),
            int(im.center[0] - width_crop_rect / 2),
        )
        x1, y1 = (
            int(im.center[1] + heigth_crop_rect / 2),
            int(im.center[0] + width_crop_rect / 2),
        )
        im_crop = im.crop_image_horizontal(x0, y0, x1, y1)

        crop_translation_vector = self.center - im_crop.center

        if return_transfo_bricks:
            return im_crop, translation_vector, angle, crop_translation_vector
        else:
            return im_crop

    def save(self, save_filepath: str):
        self.show(save_filepath=save_filepath)
