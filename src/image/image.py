"""
Image module for image processing
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import cv2
import matplotlib.pyplot as plt
import src.geometry as geo
from src.image.utils.render import PolygonsRender, SegmentsRender
from src.image.drawer import DrawerImage
from src.image.transformer import TransformerImage
from src.image.reader import ReaderImage


class Image(ReaderImage, DrawerImage, TransformerImage):
    """Image class"""

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

    def save(self, save_filepath: str) -> None:
        """Save the image in a local file

        Args:
            save_filepath (str): path to the file
        """
        self.show(save_filepath=save_filepath)

    def iou(self, other: Image) -> float:
        """Compute the intersection over union score

        Args:
            other (Image): another image

        Returns:
            float: a score from 0 to 1. The greater the score the greater the other
                image is equal to the original image
        """
        assert self.is_equal_shape(other)
        mask0 = self.binaryrev()
        mask1 = other.binaryrev()
        return np.sum(mask0 * mask1) / np.count_nonzero(mask0 + mask1)

    def score_contains(
        self, other: Image, binarization_method: str = "sauvola"
    ) -> float:
        """How much the other image is contained in the original image

        Args:
            other (Image): other Image object
            method (str, optional): binarization method. Defaults to "sauvola".

        Returns:
            float: a score from 0 to 1. The greater the score the greater the other
                image is contained within the original image
        """
        assert self.is_equal_shape(other, consider_channel=False)
        other_binaryrev = other.binaryrev(method=binarization_method)
        return np.sum(
            self.binaryrev(method=binarization_method) * other_binaryrev
        ) / np.sum(other_binaryrev)

    def score_contains_polygons(
        self,
        polygons: list[geo.Polygon],
        dilate_kernel: tuple = (5, 5),
        dilate_iterations: int = 1,
        binarization_method: str = "sauvola",
        resize_factor: float = 1.0,
    ) -> np.ndarray:
        """Check how much polygons are contained in the image

        Beware: this method is different from the score_contains method because in
        this case you can emphasize the base image by dilating its content.

        Everything that is a 1 in the rmask will be dilated to give more chance for the
        contour to be contained within the image in the calculation. This way you
        can control the sensitivity of the score.

        Args:
            polygon (Polygon): Polygon object
            dilate_kernel (tuple, optional): dilate kernel param. Defaults to (5, 5).
            dilate_iterations (int, optional): dilate iterations param. Defaults to 1.
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
        scores = []
        for polygon in polygons:

            im = self.crop_polygon(polygon=polygon, copy=True)

            im.dilate(kernel=dilate_kernel, iterations=dilate_iterations)

            im.resize(factor=resize_factor)

            # create all-white image of same size as original with the geometry entity
            other = (
                im.copy()
                .as_white()
                .draw_polygons(
                    polygons=[polygon],
                    render=PolygonsRender(thickness=1, default_color=(0, 0, 0)),
                )
                .as_grayscale()
            )

            cur_score = im.score_contains(
                other=other, binarization_method=binarization_method
            )

            scores.append(cur_score)

        return np.asarray(scores)

    def score_contains_segments(
        self,
        segments: list[geo.Segment],
        dilate_kernel: tuple = (5, 5),
        dilate_iterations: int = 1,
        binarization_method: str = "sauvola",
        resize_factor: float = 1.0,
    ) -> np.ndarray:
        """Compute the contain score for each individual segment.
        This method can be better than :func:`~Image.score_contains_polygons()` in some
        cases.
        It provides a score for each single segments. This way it is better to
        identify which segments specifically are contained in the image or not.

        Args:
            segments (np.ndarray | list[geo.Segment]): a list of segments
            dilate_kernel (tuple, optional): dilate kernel param. Defaults to (5, 5).
            dilate_iterations (int, optional): dilate iterations param. Defaults to 1.
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
        score_segments = []

        for segment in segments:

            im = self.crop_around_segment_horizontal_faster(
                segment=segment.asarray,
                dim_crop_rect=(-1, height_crop),
                added_width=added_width,
            )

            im.dilate(kernel=dilate_kernel, iterations=dilate_iterations)

            im.resize(factor=resize_factor)

            segment_crop = geo.Segment(
                np.array(
                    [
                        [added_width, mid_height_crop],
                        [segment.length + added_width, mid_height_crop],
                    ]
                )
                * resize_factor
            )

            other = (
                im.copy()
                .as_white()
                .draw_segments(
                    segments=[segment_crop],
                    render=SegmentsRender(thickness=1, default_color=(0, 0, 0)),
                )
                .as_grayscale()
            )

            score = im.score_contains(
                other=other, binarization_method=binarization_method
            )

            score_segments.append(score)

        return np.asarray(score_segments)

    def score_distance_from_center(
        self, point: np.ndarray, method: str = "linear"
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
        valid_score_dist_methods = ["linear", "gaussian"]

        def gaussian_2d(
            x: float,
            y: float,
            x0: float = 0.0,
            y0: float = 0.0,
            amplitude: float = 1,
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
            amplitude: float = 1,
            radius: float = 1,
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

        raise ValueError(
            f"The method {method} should be in the valid methods"
            f"{valid_score_dist_methods}"
        )

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
        xmax = min(self.width, xmax)
        ymax = min(self.height, ymax)

        # recreate a rectangle with new coordinates
        rect_restricted = geo.Rectangle.from_topleft_bottomright(
            topleft=np.asarray([xmin, ymin]),
            bottomright=np.asarray([xmax, ymax]),
            is_cast_int=True,
        )
        return rect_restricted
