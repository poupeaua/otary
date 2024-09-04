"""
Image Drawer module. It only contains methods to draw objects in images.
"""

from __future__ import annotations

from typing import Self
from abc import ABC
import cv2
import numpy as np

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
from src.image.base import BaseImage


class DrawerImage(BaseImage, ABC):
    """Image Drawer class to draw objects on a given image"""

    def __pre_draw(self, n_objects: int, render: Render) -> np.ndarray:
        render.adjust_colors_length(n=n_objects)
        return self.as_colorscale().asarray

    def draw_points(
        self,
        points: np.ndarray | list[geo.Point],
        render: PointsRender = PointsRender(),
    ) -> Self:
        """Add points in the image

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
        self.asarray = im_array
        return self

    def draw_segments(
        self,
        segments: np.ndarray | list[geo.Segment],
        render: SegmentsRender = SegmentsRender(),
    ) -> Self:
        """Add segments in the image. It can be arrowed segments (vectors) too.

        Args:
            segments (np.ndarray): list of segments

        Returns:
            (DrawerImage): original image changed that contains the segments drawn
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
        self.asarray = im_array
        return self

    def draw_contours(
        self, contours: list[geo.Contour], render: ContoursRender = ContoursRender()
    ) -> Self:
        """Add contours in the image

        Args:
            contours (list[geo.Contour]): list of Contour objects

        Returns:
            DrawerImage: image with the added contours
        """
        assert isinstance(render, ContoursRender)
        for cnt in contours:
            self.draw_segments(segments=cnt.lines, render=render)
        return self

    def draw_ocr_outputs(
        self,
        ocr_outputs: list[OcrSingleOutput],
        render: OcrSingleOutputRender = OcrSingleOutputRender(),
    ) -> Self:
        """Return the image with the bounding boxes displayed from a list of OCR
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
        self.asarray = im_array
        return self
