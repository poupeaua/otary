"""
Image Drawer module. It only contains methods to draw objects in images.
"""

from __future__ import annotations

from typing import Self
from abc import ABC
import cv2
import numpy as np

import src.geometry as geo
from src.cv.ocr.output.ocr_single_output import OcrSingleOutput
from src.image.utils.tools import prep_obj_draw
from src.image.utils.render import (
    Render,
    PointsRender,
    CirclesRender,
    SegmentsRender,
    LinearSplinesRender,
    PolygonsRender,
    OcrSingleOutputRender,
)
from src.image.base import BaseImage


class DrawerImage(BaseImage, ABC):
    """Image Drawer class to draw objects on a given image"""

    def _pre_draw(self, n_objects: int, render: Render) -> np.ndarray:
        render.adjust_colors_length(n=n_objects)
        return self.as_colorscale().asarray

    def draw_circles(
        self,
        circles: list[geo.Circle],
        render: CirclesRender = CirclesRender(),
    ) -> Self:
        """Draw circles in the image

        Args:
            circles (list[Circle]): list of Circle geometry objects.

        Returns:
            Image: new image
        """
        im_array = self._pre_draw(n_objects=len(circles), render=render)
        for circle, color in zip(circles, render.colors):
            cv2.circle(  # type: ignore[call-overload]
                img=im_array,
                center=np.round(circle.center).astype(int),
                radius=int(circle.radius),
                color=color,
                thickness=render.thickness,
                lineType=render.line_type,
            )
            if render.is_draw_center_point_enabled:
                cv2.circle(  # type: ignore[call-overload]
                    img=im_array,
                    center=np.round(circle.center).astype(int),
                    radius=1,
                    color=color,
                    thickness=render.thickness,
                    lineType=render.line_type,
                )
        self.asarray = im_array
        return self

    def draw_points(
        self,
        points: np.ndarray | list[geo.Point],
        render: PointsRender = PointsRender(),
    ) -> Self:
        """Draw points in the image

        Args:
            points (np.ndarray): list of points. It must be of shape (n, 2). This
                means n points of shape 2 (x and y coordinates).

        Returns:
            Image: new image
        """
        _points = prep_obj_draw(objects=points, _type=geo.Point)
        im_array = self._pre_draw(n_objects=len(_points), render=render)
        for point, color in zip(_points, render.colors):
            cv2.circle(
                img=im_array,
                center=point,
                radius=render.radius,
                color=color,
                thickness=render.thickness,
                lineType=render.line_type,
            )
        self.asarray = im_array
        return self

    def draw_segments(
        self,
        segments: np.ndarray | list[geo.Segment],
        render: SegmentsRender = SegmentsRender(),
    ) -> Self:
        """Draw segments in the image. It can be arrowed segments (vectors) too.

        Args:
            segments (np.ndarray): list of segments. Can be a numpy array of shape
                (n, 2, 2) which means n array of shape (2, 2) that define a segment
                by two 2D points.

        Returns:
            (DrawerImage): original image changed that contains the segments drawn
        """
        _segments = prep_obj_draw(objects=segments, _type=geo.Segment)
        im_array = self._pre_draw(n_objects=len(segments), render=render)
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
            for segment, color in zip(_segments, render.colors):
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

    def draw_splines(
        self,
        splines: list[geo.LinearSpline],
        render: LinearSplinesRender = LinearSplinesRender(),
    ) -> Self:
        """Draw linear splines in the image.

        Args:
            splines (list[geo.LinearSpline]): linear splines to draw.
            render (LinearSplinesRender, optional): linear splines render.
                Defaults to LinearSplinesRender().

        Returns:
            Self: image with the added splines drawn.
        """
        _splines = prep_obj_draw(objects=splines, _type=geo.LinearSpline)
        im_array = self._pre_draw(n_objects=len(_splines), render=render)
        for spline, color in zip(_splines, render.colors):

            if render.as_vectors:
                cv2.polylines(
                    img=im_array,
                    pts=[spline[:-1]],
                    isClosed=False,
                    color=color,
                    thickness=render.thickness,
                    lineType=render.line_type,
                )

                # Draw the last edge as a vector
                ix = int(len(spline) * (1 - render.pct_ix_head))
                ix = ix - 1 if ix == len(spline) - 1 else ix
                segment = [spline[ix], spline[-1]]
                cv2.arrowedLine(
                    img=im_array,
                    pt1=segment[0],
                    pt2=segment[1],
                    color=color,
                    thickness=render.thickness,
                    tipLength=render.tip_length / geo.Segment(segment).length,
                )

            else:
                cv2.polylines(
                    img=im_array,
                    pts=[spline],
                    isClosed=False,
                    color=color,
                    thickness=render.thickness,
                    lineType=render.line_type,
                )
        return self

    def draw_polygons(
        self, polygons: list[geo.Polygon], render: PolygonsRender = PolygonsRender()
    ) -> Self:
        """Draw polygons in the image

        Args:
            polygons (list[Polygon]): list of Polygon objects
            render (PolygonsRender): PolygonRender object

        Returns:
            Image: image with the added polygons
        """
        _polygons = prep_obj_draw(objects=polygons, _type=geo.Polygon)
        im_array = self._pre_draw(n_objects=len(_polygons), render=render)
        for polygon, color in zip(_polygons, render.colors):
            if render.is_filled:
                cv2.fillPoly(
                    img=im_array,
                    pts=[polygon],
                    color=color,
                    lineType=render.line_type,
                )
            else:
                cv2.polylines(
                    img=im_array,
                    pts=[polygon],
                    isClosed=True,
                    color=color,
                    thickness=render.thickness,
                    lineType=render.line_type,
                )
        self.asarray = im_array
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
            ocr_outputs (list[OcrSingleOutput]): list of OcrSingleOutput objects

        Returns:
            Image: a new image with the bounding boxes displayed
        """
        im_array = self._pre_draw(n_objects=len(ocr_outputs), render=render)
        for ocrso, color in zip(ocr_outputs, render.colors):
            if not isinstance(ocrso, OcrSingleOutput) or ocrso.bbox is None:
                # warnings.warn(
                #     f"Object {ocrso} is not an OcrSingleOutput or has no bbox. "
                #     "Skipping it."
                # )
                continue
            cnt = [ocrso.bbox.asarray.reshape((-1, 1, 2)).astype(np.int32)]
            im_array = cv2.drawContours(
                image=im_array,
                contours=cnt,
                contourIdx=-1,
                thickness=render.thickness,
                color=color,
                lineType=render.line_type,
            )
        self.asarray = im_array
        return self
