"""
Image manipulation module
"""

from __future__ import annotations

from typing import Optional

from abc import ABC
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage

import src.geometry as geo
from src.core.dataclass.ocrsingleoutput import OcrSingleOutput
from src.image.tools import prep_obj_draw
from src.image.render import (
    Render,
    PointsRender,
    SegmentsRender,
    ContoursRender,
    OcrSingleOutputRender,
)


class BaseImage(ABC):
    """Base Image class"""

    def __init__(self, image: np.ndarray | Image | BaseImage) -> None:
        if isinstance(image, (Image, BaseImage)):
            image = image.asarray
        self.__asarray: np.ndarray = image.copy()

    @property
    def baseimage(self) -> BaseImage:
        """BaseImage object"""
        return self

    @property
    def asarray(self) -> np.ndarray:
        """Array representation of the image"""
        return self.__asarray

    @property
    def is_gray(self) -> bool:
        """Whether the image is a grayscale image or not

        Returns:
            bool: True if image is in grayscale, 0 otherwise
        """
        return bool(len(self.asarray.shape) == 2)

    @property
    def shape(self) -> tuple:
        """Returns the image shape value

        Returns:
            tuple[int]: image shape
        """
        return self.asarray.shape

    @property
    def height(self) -> int:
        """Height of the image. In cv2 it is defined as the first image shape value

        Returns:
            int: image height
        """
        return self.asarray.shape[0]

    @property
    def width(self) -> int:
        """Width of the image. In cv2 it is defined as the second image shape value

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
    def center(self) -> np.ndarray:
        """Center of the image.

        Please note that it is returned as type int because the center needs to
        represent a X-Y coords of a pixel.

        Returns:
            np.ndarray: center point of the image
        """
        return (np.array([self.width, self.height]) / 2).astype(int)

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
    def asarray_norm(self) -> np.ndarray:
        """Returns the representation of the image as a array with value not in
        [0, 255] but in [0, 1].

        Returns:
            np.ndarray: an array with value in [0, 1]
        """
        return self.asarray / 255

    @property
    def as_grayscale(self) -> Image:
        """Generate the image in grayscale of shape (height, width)

        Returns:
            Image: original image in grayscale
        """
        if self.is_gray:
            return self
        return Image(cv2.cvtColor(self.asarray, cv2.COLOR_BGR2GRAY))

    @property
    def as_colorscale(self) -> Image:
        """Generate the image in colorscale (height, width, 3).
        This property can be useful when we wish to draw objects in a given color
        on a grayscale image.

        Returns:
            Image: original image in color
        """
        if not self.is_gray:
            return self
        return Image(cv2.cvtColor(self.asarray, cv2.COLOR_GRAY2BGR))

    @property
    def as_white(self) -> Image:
        """Returns an entirely white image with the same dimension as the original.

        Returns:
            Image: new white image
        """
        return self.as_filled(fill_value=255)

    @property
    def corners(self) -> np.ndarray:
        """Returns the corners in the following order:

        0. bottom left corner
        1. bottom right corner
        2. top right corner
        3. top left corner

        Returns:
            np.ndarray: array containing the corners
        """
        return np.array(
            [[0, 0], [self.width, 0], [self.width, self.height], [0, self.height]]
        )

    def is_equal_shape(self, other: Image) -> bool:
        """Check whether two images have the same shape

        Args:
            other (Image): Image object

        Returns:
            bool: True if the objects have the same shape, False otherwise
        """
        return self.shape == other.shape

    def dist_pct(self, pct: float = 0.01) -> float:
        """Distance percentage that can be used an acceptable distance error margin.
        It is calculated based on the normalized side length.

        Args:
            pct (float, optional): pourcentage of distance error. Defaults to 0.01,
                which means 1% of the normalized side length as the
                default margin distance error.

        Returns:
            float: margin distance error
        """
        return self.norm_side_length * pct

    def as_filled(self, fill_value: int | np.ndarray = 255) -> Image:
        """Returns an entirely white image of the same size as the original.
        Can be useful to get an empty representation of the same image to paint
        and draw things on an image of the same dimension.

        Args:
            fill_value (int | np.ndarray, optional): color to fill the new empty image.
                Defaults to 255 which means that is returns a entirely white image.

        Returns:
            Image: new image with a single color of the same size as original.
        """
        return Image(np.full(shape=self.shape, fill_value=fill_value, dtype=np.uint8))

    def copy(self) -> Image:
        """Copy of the image

        Returns:
            Image: image copy
        """
        return Image(self.asarray.copy())


class DrawerImage(BaseImage, ABC):
    """Image Drawer class to draw objects on a given image"""

    def __pre_draw(self, n_objects: int, render: Render) -> np.ndarray:
        im = self.asarray.copy()

        if Image(im).is_gray:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

        render.adjust_colors_length(n=n_objects)
        return im

    def draw_points(
        self,
        points: np.ndarray | list[geo.Point],
        render: PointsRender = PointsRender(),
    ) -> Image:
        """Add points on an image

        Args:
            points (np.ndarray): points

        Returns:
            Image: new image
        """
        assert isinstance(render, PointsRender)
        _points = prep_obj_draw(objects=points, _type=geo.Point)
        im_array = self.__pre_draw(n_objects=len(_points), render=render)
        for point, color in zip(_points, render.colors):
            cv2.circle(
                img=im_array,
                center=point,
                radius=render.radius,
                color=color,
                thickness=render.thickness,
            )
        return Image(im_array)

    def draw_segments(
        self,
        segments: np.ndarray | list[geo.Segment],
        render: SegmentsRender = SegmentsRender(),
    ) -> Image:
        """Add segments on an image. It can be arrowed segments (vectors) too.

        Args:
            segments (np.ndarray): list of segments

        Returns:
            (Image): a new image that contains the segments drawn
        """
        assert isinstance(render, SegmentsRender)
        _segments = prep_obj_draw(objects=segments, _type=geo.Segment)
        im_array = self.__pre_draw(n_objects=len(segments), render=render)
        if render.as_vectors:
            for segment, color in zip(_segments, render.colors):
                cv2.arrowedLine(
                    img=im_array,
                    pt1=segment[0],
                    pt2=segment[1],
                    color=color,
                    thickness=render.thickness,
                    line_type=render.line_type,
                    tipLength=render.tip_length / geo.Segment(segment).length,
                )
        else:
            for segment, color in zip(segments, render.colors):
                cv2.line(
                    img=im_array,
                    pt1=segment[0],
                    pt2=segment[1],
                    color=color,
                    thickness=render.thickness,
                    lineType=render.line_type,
                )
        return Image(im_array)

    def draw_contours(
        self, contours: list[geo.Contour], render: ContoursRender = ContoursRender()
    ) -> Image:
        """Add a contour in a image

        Args:
            contours (list[geo.Contour]): list of Contour objects

        Returns:
            Image: image with the added contours
        """
        assert isinstance(render, ContoursRender)
        im = self.copy()
        for cnt in contours:
            im = im.draw_segments(segments=cnt.lines, render=render)

        return Image(im)

    def draw_ocr_outputs(
        self,
        ocr_outputs: list[OcrSingleOutput],
        render: OcrSingleOutputRender = OcrSingleOutputRender(),
    ) -> Image:
        """Return a new image with the bounding boxes displayed from a list of OCR
        single output. It allows you to show bounding boxes that can have an angle,
        not necessarily vertical or horizontal.

        Args:
            ocr_outputs (list[OcrSingleOutput]): list of OcrSingleOutput dataclass.

        Returns:
            (Image): a new image with the bounding boxes displayed
        """
        assert isinstance(render, OcrSingleOutputRender)
        im_array = self.__pre_draw(n_objects=len(ocr_outputs), render=render)
        for ocrso, color in zip(ocr_outputs, render.colors):
            cnt = [ocrso.bbox.asarray.reshape((-1, 1, 2)).astype(np.int32)]
            im_array = cv2.drawContours(
                image=im_array,
                contours=cnt,
                contourIdx=-1,
                thickness=render.thickness,
                color=color,
            )
        return Image(im_array)


class TransformerImage(BaseImage, ABC):
    """Transform images utility class"""

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
        self, point: np.ndarray, mode: str = "constant"
    ) -> tuple[Image, np.ndarray]:
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
        return im, shift_vector

    def center_image_to_segment(
        self, segment: np.ndarray, mode: str = "constant"
    ) -> tuple[Image, np.ndarray]:
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
            point=geo.Segment(segment).centroid, mode=mode
        )

    def resize_fixed(
        self, dim: tuple[int, int], interpolation: int = cv2.INTER_AREA
    ) -> Image:
        """Resize the image using a fixed dimension well defined.
        This function can result in a distorted image.

        Args:
            dim (tuple[int, int]): a tuple with two integer, width, height.
            interpolation (int, optional): resize interpolation.
                Defaults to cv2.INTER_AREA.

        Returns:
            Image: image object
        """
        im = self.asarray.copy()

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

        im = cv2.resize(src=im, dsize=_dim, interpolation=interpolation)
        return Image(im)

    def resize(
        self, scale_percent: float, interpolation: int = cv2.INTER_AREA
    ) -> Image:
        """Resize the image to a new size using a scaling percentage value that
        will be applied to all dimensions (width and height).
        Applying this method can not result in a distorted image.

        Args:
            scale_percent (float): scale to resize the image. A value 100 does not
                change the image. 200 double the image size.

        Returns:
            (Image): new resized image
        """
        if scale_percent == 100:
            return Image(self.baseimage)

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
        dim_crop_rect: tuple[int, int] = (-1, 100),
        default_extra_width: int = 75,
    ) -> tuple[Image, np.ndarray, float, np.ndarray]:
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
            tuple[Image, np.ndarray, float]: _description_
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


class AnalyzerImage(BaseImage, ABC):
    """AnalyzerImage class used for extracting information of the image"""

    def adaptative_thresholding(
        self, ksize: int = 5, adaptative_method: int = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    ) -> Image:
        """Apply Adaptative thresholding.
        A blur is applied before for better masking results.
        See https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html

        As the input image must be a grayscale before applying any thresholding
        methods we convert the image to grayscale.

        Returns:
            Image: image thresholded where its values are now pure 0 or 255
        """
        blur = cv2.medianBlur(src=self.as_grayscale.asarray, ksize=ksize)
        masked_img = cv2.adaptiveThreshold(
            src=blur,
            maxValue=255,
            adaptiveMethod=adaptative_method,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=11,
            C=2,
        )
        return Image(masked_img)

    def otsu_thresholding(self, ksize: int = 5) -> Image:
        """Apply Ostu thresholding. A blur is applied before for better masking results.
        See https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html.

        As the input image must be a grayscale before applying any thresholding
        methods we convert the image to grayscale.

        Returns:
            Image: image thresholded where its values are now pure 0 or 255
        """
        blur = cv2.GaussianBlur(
            src=self.as_grayscale.asarray, ksize=(ksize, ksize), sigmaX=0
        )
        _, masked_img = cv2.threshold(
            src=blur, thresh=0, maxval=255, type=cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return Image(masked_img)

    @property
    def mask(self) -> np.ndarray:
        """Mask representation of the image with values that can be only 0 or 1.
        The value 0 is now 0 and value of 255 are now 1. Black is 0 and white is 1.

        Returns:
            np.ndarray: array where its inner values are 0 or 1
        """
        return self.otsu_thresholding().asarray_norm.astype(np.uint8)

    @property
    def rmask(self) -> np.ndarray:
        """Reversed mask of the image.
        The value 0 is now 1 and value of 255 are now 0. Black is 1 and white is 0.

        Returns:
            np.ndarray: array where its inner values are 0 or 1
        """
        return 1 - self.mask

    def iou(self, other: Image) -> float:
        """Compute the intersection over union score

        Args:
            other (Image): another image

        Returns:
            float: a score from 0 to 1. The greater the score the greater the other
                image is equal to the original image
        """
        assert self.is_equal_shape(other)
        mask0 = self.mask
        mask1 = other.mask
        return 2 * (mask0 * mask1) / (mask0 + mask1)

    def score_contains(self, other: Image) -> float:
        """How much the other image is contained in the original image

        Args:
            other (Image): other Image object

        Returns:
            float: a score from 0 to 1. The greater the score the greater the other
                image is contained within the original image
        """
        assert self.is_equal_shape(other)
        other_rmask = other.rmask
        return np.sum(self.rmask * other_rmask) / np.sum(other_rmask)

    def score_contains_contour(
        self,
        contour: geo.Contour,
        dilate_kernel: tuple = (5, 5),
        dilate_iterations: int = 1,
    ) -> float:
        """Check how much the contour is contained in the original image.

        Beware: this method is different from the score_contains method because in
        this case we emphasize the base image by dilating its content.
        Everything that is a 1 in the rmask will be dilated to give more chance for the
        contour to be contained within the image in the calculation.

        Args:
            contour (Contour): Contour object

        Returns:
            float: a score from 0 to 1. The greater the score the greater the contour
                 is contained within the original image
        """
        # create all-white image of same size as original with the geometry entity
        other = self.as_white.draw_contours(
            contours=[contour],
            render=ContoursRender(thickness=1, default_color=(0, 0, 0)),
        ).as_grayscale

        # dilate the original image
        im = Image(
            (
                1
                - cv2.dilate(
                    self.rmask,
                    kernel=np.ones(dilate_kernel, np.uint8),
                    iterations=dilate_iterations,
                )
            )
            * 255
        )
        return im.score_contains(other=other)

    def score_distance_from_center(
        self, point: np.ndarray, sigma: float = 1.0, method: str = "gaussian"
    ) -> float:
        """Compute a score to evaluate how far a point is from the
        image center point. A score equal to 1 means that the contour and image centers
        coincide.

        A score close to 0 means that the point and the image center are far away.
        A score close to 1 means that the point and the image center are close.

        This method can be used to compute a score for a contour centroid:
        - A small score should be taken into account and informs us that the contour
        found is probably wrong.
        - On the contrary, a high score does not ensure a high quality contour.

        Args:
            point (np.ndarray): 2D point
            sigma (float, optional): the standard variation for the
                score evaluation function. Defaults to 1.0.

        Returns:
            float: a score from 0 to 1.
        """

        def gaussian2D(
            x: float,
            y: float,
            x0: float = 0.0,
            y0: float = 0.0,
            amplitude: float = 1,
            sigmax: float = 1.0,
            sigmay: float = 1.0,
        ) -> float:
            return amplitude * np.exp(
                -((x - x0) ** 2) / (2 * sigmax) - (y - y0) ** 2 / (2 * sigmay)
            )

        if method == "gaussian":
            return gaussian2D(
                x=point[0],
                y=point[1],
                x0=self.center[0],
                y0=self.center[1],
                sigmax=sigma,
                sigmay=sigma,
            )
        else:
            raise NotImplementedError(f"The method {method} is not implemented.")


class Image(DrawerImage, TransformerImage, AnalyzerImage):
    """Image Manipulation class"""

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
        else:
            im = self.asarray

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

    def save(self, save_filepath: str):
        """Save the image in a local file

        Args:
            save_filepath (str): path to the file
        """
        self.show(save_filepath=save_filepath)
