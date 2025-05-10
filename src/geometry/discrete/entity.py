"""
DiscreteGeometryEntity module class
"""

from __future__ import annotations

from typing import Optional, Self, TYPE_CHECKING
import copy
from abc import ABC, abstractmethod
from shapely import (
    GeometryCollection,
)

import cv2
import numpy as np

from src.geometry.utils.tools import rotate_2d_points
from src.geometry.entity import GeometryEntity
from src.utils.tools import assert_transform_shift_vector

if TYPE_CHECKING:
    from src.geometry import Polygon, Rectangle, Segment  # pragma: no cover


class DiscreteGeometryEntity(GeometryEntity, ABC):
    """GeometryEntity class which is the abstract base class for all geometry classes"""

    def __init__(self, points, is_cast_int: bool = False) -> None:
        _arr = (
            np.asarray(points)
            if not is_cast_int
            else np.round(np.asarray(points)).astype(int)
        )
        self.points = copy.deepcopy(_arr)
        self.is_cast_int = is_cast_int

    # --------------------------------- PROPERTIES ------------------------------------

    @property
    @abstractmethod
    def shapely_edges(self) -> GeometryCollection:
        """Representation of the geometric object in the shapely library
        as a geometrical object defined only as a curve with no area. Particularly
        useful to look for points intersections
        """

    @property
    @abstractmethod
    def shapely_surface(self) -> GeometryCollection:
        """Representation of the geometric object in the shapely library
        as a geometrical object with an area and a border. Particularly useful
        to check if two geometrical objects are contained within each other or not.
        """

    @property
    @abstractmethod
    def edges(self) -> np.ndarray:
        """Get the edges of the geometry entity

        Returns:
            np.ndarray: edges of the geometry entity
        """

    @property
    def segments(self) -> list[Segment]:
        """Get the segments of the geometry entity

        Returns:
            np.ndarray: segments of the geometry entity
        """
        # pylint: disable=import-outside-toplevel
        from src.geometry import Segment  # delayed import to avoid circular import

        return [Segment(e) for e in self.edges]

    @property
    def n_points(self) -> int:
        """Returns the number of points this geometric object is made of

        Returns:
            int: number of points that composes the geomtric object
        """
        return self.points.shape[0]

    @property
    def asarray(self) -> np.ndarray:
        """Array representation of the geometry object"""
        return self.points

    @asarray.setter
    def asarray(self, value: np.ndarray):
        """Setter for the asarray property

        Args:
            value (np.ndarray): value of the asarray to be changed
        """
        self.points = value

    @property
    def centroid(self) -> np.ndarray:
        """Compute the centroid point which can be seen as the center of gravity of
        the shape

        Returns:
            np.ndarray: centroid point
        """
        return np.mean(self.points, axis=0)

    @property
    def xmax(self) -> float:
        """Get the maximum X coordinate of the geometry entity

        Returns:
            np.ndarray: 2D point
        """
        return np.max(self.asarray[:, 0])

    @property
    def xmin(self) -> float:
        """Get the minimum X coordinate of the geometry entity

        Returns:
            np.ndarray: 2D point
        """
        return np.min(self.asarray[:, 0])

    @property
    def ymax(self) -> float:
        """Get the maximum Y coordinate of the geometry entity

        Returns:
            np.ndarray: 2D point
        """
        return np.max(self.asarray[:, 1])

    @property
    def ymin(self) -> float:
        """Get the minimum Y coordinate of the geometry entity

        Returns:
            np.ndarray: 2D point
        """
        return np.min(self.asarray[:, 1])

    @property
    def lengths(self) -> np.ndarray:
        """Returns the length of all the segments that make up the Polygon

        Returns:
            np.ndarray: array of shape (n_points)
        """
        lengths: np.ndarray = np.linalg.norm(np.diff(self.edges, axis=1), axis=2)
        return lengths.flatten()

    @property
    def crop_coordinates(self) -> np.ndarray:
        """Compute the coordinates of the geometry entity in the context of
        itself being in a crop image that make it fit pefectly

        Returns:
            Self: _description_
        """
        return self.asarray - np.array([self.xmin, self.ymin])

    # ---------------------------- MODIFICATION METHODS -------------------------------

    def rotate(
        self,
        angle: float,
        is_degree: bool = False,
        is_clockwise: bool = True,
        pivot: Optional[np.ndarray] = None,
    ) -> Self:
        """Rotate the geometry entity object.
        A pivot point can be passed as an argument to rotate the object around the pivot

        Args:
            angle (float): rotation angle
            is_degree (bool, optional): whether the angle is in degree or radian.
                Defaults to False which means radians.
            is_clockwise (bool, optional): whether the rotation is clockwise or
                counter-clockwise. Defaults to True.
            pivot (np.ndarray, optional): pivot point.
                Defaults to None which means that by default the centroid point of
                the shape is taken as the pivot point.

        Returns:
            GeometryEntity: rotated geometry entity object.
        """
        if pivot is None:
            pivot = self.centroid

        self.points = rotate_2d_points(
            points=self.points,
            angle=angle,
            pivot=pivot,
            is_degree=is_degree,
            is_clockwise=is_clockwise,
        )
        return self

    def rotate_around_image_center(
        self, img: np.ndarray, angle: float, degree: bool = False
    ) -> Self:
        """Given an geometric object and an image, rotate the object around
        the image center point.

        Args:
            img (np.ndarray): image as a shape (x, y) sized array
            angle (float): rotation angle
            degree (bool, optional): whether the angle is in degree or radian.
                Defaults to False which means radians.

        Returns:
            GeometryEntity: rotated geometry entity object.
        """
        img_center_point = np.array([img.shape[1], img.shape[0]]) / 2
        return self.rotate(angle=angle, pivot=img_center_point, is_degree=degree)

    def shift(self, vector: np.ndarray) -> Self:
        """Shift the geometry entity by the vector direction

        Args:
            vector (np.ndarray): vector that describes the shift as a array with
                two elements. Example: [2, -8] which describes the
                vector [[0, 0], [2, -8]]. The vector can also be a vector of shape
                (2, 2) of the form [[2, 6], [1, 3]].

        Returns:
            GeometryEntity: shifted geometrical object
        """
        vector = assert_transform_shift_vector(vector=vector)
        self.points = self.points + vector
        return self

    def clamp(
        self,
        xmin: float = -np.inf,
        xmax: float = np.inf,
        ymin: float = -np.inf,
        ymax: float = np.inf,
    ) -> Self:
        """Clamp the Geometry entity so that the x and y coordinates fit in the
        min and max values in parameters.

        Args:
            xmin (float): x coordinate minimum
            xmax (float): x coordinate maximum
            ymin (float): y coordinate minimum
            ymax (float): y coordinate maximum

        Returns:
            GeometryEntity: clamped GeometryEntity
        """
        self.asarray[:, 0] = np.clip(self.asarray[:, 0], xmin, xmax)  # x values
        self.asarray[:, 1] = np.clip(self.asarray[:, 1], ymin, ymax)  # y values
        return self

    def normalize(self, x: float, y: float) -> Self:
        """Normalize the geometry entity by dividing the points by a norm on the
        x and y coordinates.

        Args:
            x (float): x coordinate norm
            y (float): y coordinate norm

        Returns:
            GeometryEntity: normalized GeometryEntity
        """
        if x == 0 or y == 0:
            raise ValueError("x or y cannot be 0")
        self.asarray = self.asarray / np.array([x, y])
        return self

    # ------------------------------- CLASSIC METHODS ---------------------------------

    def copy(self) -> Self:
        """Create a copy of the geometry entity object

        Returns:
            GeometryEntity: copy of the geometry entity object
        """
        return type(self)(points=self.asarray.copy(), is_cast_int=self.is_cast_int)

    def enclosing_axis_aligned_bbox(self) -> Rectangle:
        """Compute the smallest area enclosing Axis-Aligned Bounding Box (AABB)
        See: https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html

        Returns:
            Rectangle: Rectangle object
        """
        # pylint: disable=import-outside-toplevel
        from src.geometry import Rectangle  # delayed import to avoid circular import

        topleft_x, topleft_y, width, height = cv2.boundingRect(
            array=self.asarray.astype(np.float32)
        )
        bbox = np.array(
            [
                [topleft_x, topleft_y],
                [topleft_x + width, topleft_y],
                [topleft_x + width, topleft_y + height],
                [topleft_x, topleft_y + height],
            ]
        )
        return Rectangle(bbox)

    def enclosing_oriented_bbox(self) -> Rectangle:
        """Compute the smallest area enclosing Oriented Bounding Box (OBB)
        See: https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html

        Returns:
            Rectangle: Rectangle object
        """
        # pylint: disable=import-outside-toplevel
        from src.geometry import Rectangle  # delayed import to avoid circular import

        rect = cv2.minAreaRect(self.asarray)
        bbox = cv2.boxPoints(rect)
        return Rectangle(bbox)

    def enclosing_convex_hull(self) -> Polygon:
        """Compute the smallest area enclosing Convex Hull
        See: https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html

        Returns:
            Polygon: Polygon object
        """
        # pylint: disable=import-outside-toplevel
        from src.geometry import Polygon  # delayed import to avoid circular import

        convexhull = np.squeeze(cv2.convexHull(self.asarray))
        return Polygon(convexhull)

    def distances_vertices_to_point(self, point: np.ndarray) -> np.ndarray:
        """Get the distance from all vertices in the geometry entity to the input point

        Args:
            point (np.ndarray): 2D point

        Returns:
            np.ndarray: array of the same len as the number of vertices in the geometry
                entity.
        """
        return np.linalg.norm(self.asarray - point, axis=1)

    def shortest_dist_vertices_to_point(self, point: np.ndarray) -> float:
        """Compute the shortest distance from the geometry entity vertices to the point

        Args:
            point (np.ndarray): 2D point

        Returns:
            float: shortest distance from the geometry entity vertices to the point
        """
        return np.min(self.distances_vertices_to_point(point=point))

    def longest_dist_vertices_to_point(self, point: np.ndarray) -> float:
        """Compute the longest distance from the geometry entity vertices to the point

        Args:
            point (np.ndarray): 2D point

        Returns:
            float: longest distance from the geometry entity vertices to the point
        """
        return np.max(self.distances_vertices_to_point(point=point))

    def find_vertice_ix_farthest_from(self, point: np.ndarray) -> int:
        """Get the index of the farthest vertice from a given point

        Args:
            point (np.ndarray): 2D point

        Returns:
            int: the index of the farthest vertice in the entity from the input point
        """
        return np.argmax(self.distances_vertices_to_point(point=point)).astype(int)

    def find_vertice_ix_closest_from(self, point: np.ndarray) -> int:
        """Get the index of the closest vertice from a given point

        Args:
            point (np.ndarray): 2D point

        Returns:
            int: the index of the closest point in the entity from the input point
        """
        return np.argmin(self.distances_vertices_to_point(point=point)).astype(int)

    def find_shared_approx_vertices_ix(
        self, other: DiscreteGeometryEntity, margin_dist_error: float = 5
    ) -> np.ndarray:
        """Compute the vertices indices from this entity that correspond to shared
        vertices with the other geometric entity.

        A vertice is considered shared if it is close enough to another vertice
        in the other geometric structure.

        Args:
            other (DiscreteGeometryEntity): other Discrete Geometry entity
            margin_dist_error (float, optional): minimum distance to have two vertices
                considered as close enough to be shared. Defaults to 5.

        Returns:
            np.ndarray: list of indices
        """
        list_index_shared_points = []
        for i, pt in enumerate(self.asarray):
            distances = np.linalg.norm(other.asarray - pt, axis=1)
            indices = np.nonzero(distances < margin_dist_error)[0].astype(int)
            if len(indices) > 0:
                list_index_shared_points.append(i)
        return np.array(list_index_shared_points).astype(int)

    def find_shared_approx_vertices(
        self, other: DiscreteGeometryEntity, margin_dist_error: float = 5
    ) -> np.ndarray:
        """Get the shared vertices between two geometric objects.

        A vertice is considered shared if it is close enough to another vertice
        in the other geometric structure.

        Args:
            other (DiscreteGeometryEntity): a DiscreteGeometryEntity object
            margin_dist_error (float, optional): the threshold to define a vertice as
                shared or not. Defaults to 5.

        Returns:
            np.ndarray: list of vertices identified as shared between the two geometric
                objects
        """
        indices = self.find_shared_approx_vertices_ix(
            other=other, margin_dist_error=margin_dist_error
        )
        return self.asarray[indices]

    def find_vertices_far_from(
        self, points: np.ndarray, min_distance: float = 5
    ) -> np.ndarray:
        """Get vertices that belongs to the geometric structure far from the points in
        parameters.

        Args:
            points (np.ndarray): input list of points
            min_distance (float, optional): the threshold to define a point as
                far enough or not from a vertice. Defaults to 5.

        Returns:
            np.ndarray: vertices that belongs to the geometric structure and that
                are far from the input points.
        """
        list_far_points = []
        for pt in self.asarray:
            distances = np.linalg.norm(points - pt, axis=1)
            indices = np.nonzero(distances < min_distance)[0].astype(int)
            if len(indices) == 0:
                list_far_points.append(pt)
        return np.array(list_far_points)

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, DiscreteGeometryEntity):
            return False
        if not type(self) == type(value):
            return False
        return np.array_equal(self.asarray, value.asarray)

    def __neg__(self) -> Self:
        self.asarray = -self.asarray
        return self

    def __add__(self, other: np.ndarray | float | int) -> Self:
        self.asarray += other
        return self

    def __sub__(self, other: np.ndarray | float | int) -> Self:
        self.asarray -= other
        return self

    def __mul__(self, other: np.ndarray | float | int) -> Self:
        try:
            self.asarray *= other
        except Exception:
            self.asarray = self.asarray.astype(float) * other
        return self

    def __truediv__(self, other: np.ndarray | float | int) -> Self:
        self.asarray = self.asarray / other
        return self

    def __len__(self) -> int:
        return self.n_points

    def __getitem__(self, index: int) -> np.ndarray:
        return self.points[index]

    def __str__(self) -> str:
        return self.__class__.__name__ + "(" + self.asarray.tolist().__str__() + ")"

    def __repr__(self) -> str:
        return self.__class__.__name__ + "(" + self.asarray.tolist().__repr__() + ")"
