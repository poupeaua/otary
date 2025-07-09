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
from numpy.typing import NDArray

from otary.geometry.utils.tools import get_shared_point_indices, rotate_2d_points
from otary.geometry.entity import GeometryEntity
from otary.utils.tools import assert_transform_shift_vector

if TYPE_CHECKING:
    from otary.geometry import Polygon, Rectangle, Segment  # pragma: no cover


class DiscreteGeometryEntity(GeometryEntity, ABC):
    """GeometryEntity class which is the abstract base class for all geometry classes"""

    def __init__(self, points, is_cast_int: bool = False) -> None:
        _arr = (
            np.asarray(points)
            if not is_cast_int
            else np.round(np.asarray(points)).astype(np.int32)
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
    def edges(self) -> NDArray:
        """Get the edges of the geometry entity

        Returns:
            NDArray: edges of the geometry entity
        """

    @property
    def segments(self) -> list[Segment]:
        """Get the segments of the geometry entity

        Returns:
            NDArray: segments of the geometry entity
        """
        # pylint: disable=import-outside-toplevel
        from otary.geometry import Segment  # delayed import to avoid circular import

        return [Segment(e) for e in self.edges]

    @property
    def n_points(self) -> int:
        """Returns the number of points this geometric object is made of

        Returns:
            int: number of points that composes the geomtric object
        """
        return self.points.shape[0]

    @property
    def asarray(self) -> NDArray:
        """Array representation of the geometry object"""
        return self.points

    @asarray.setter
    def asarray(self, value: NDArray):
        """Setter for the asarray property

        Args:
            value (NDArray): value of the asarray to be changed
        """
        self.points = value

    @property
    @abstractmethod
    def centroid(self) -> NDArray:
        """Compute the centroid point which can be seen as the center of gravity
        or center of mass of the shape

        Returns:
            NDArray: centroid point
        """

    @property
    def center_mean(self) -> NDArray:
        """Compute the center as the mean of all the points. This can be really
        different than the centroid.

        Returns:
            NDArray: center mean as a 2D point
        """
        return np.mean(self.points, axis=0)

    @property
    def xmax(self) -> float:
        """Get the maximum X coordinate of the geometry entity

        Returns:
            NDArray: 2D point
        """
        return np.max(self.asarray[:, 0])

    @property
    def xmin(self) -> float:
        """Get the minimum X coordinate of the geometry entity

        Returns:
            NDArray: 2D point
        """
        return np.min(self.asarray[:, 0])

    @property
    def ymax(self) -> float:
        """Get the maximum Y coordinate of the geometry entity

        Returns:
            NDArray: 2D point
        """
        return np.max(self.asarray[:, 1])

    @property
    def ymin(self) -> float:
        """Get the minimum Y coordinate of the geometry entity

        Returns:
            NDArray: 2D point
        """
        return np.min(self.asarray[:, 1])

    @property
    def lengths(self) -> NDArray:
        """Returns the length of all the segments that make up the geometry entity

        Returns:
            NDArray: array of shape (n_points)
        """
        lengths: NDArray = np.linalg.norm(np.diff(self.edges, axis=1), axis=2)
        return lengths.flatten()

    @property
    def crop_coordinates(self) -> NDArray:
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
        pivot: Optional[NDArray] = None,
    ) -> Self:
        """Rotate the geometry entity object.
        A pivot point can be passed as an argument to rotate the object around the pivot

        Args:
            angle (float): rotation angle
            is_degree (bool, optional): whether the angle is in degree or radian.
                Defaults to False which means radians.
            is_clockwise (bool, optional): whether the rotation is clockwise or
                counter-clockwise. Defaults to True.
            pivot (NDArray, optional): pivot point.
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
        self, img: NDArray, angle: float, degree: bool = False
    ) -> Self:
        """Given an geometric object and an image, rotate the object around
        the image center point.

        Args:
            img (NDArray): image as a shape (x, y) sized array
            angle (float): rotation angle
            degree (bool, optional): whether the angle is in degree or radian.
                Defaults to False which means radians.

        Returns:
            GeometryEntity: rotated geometry entity object.
        """
        img_center_point = np.array([img.shape[1], img.shape[0]]) / 2
        return self.rotate(angle=angle, pivot=img_center_point, is_degree=degree)

    def shift(self, vector: NDArray) -> Self:
        """Shift the geometry entity by the vector direction

        Args:
            vector (NDArray): vector that describes the shift as a array with
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

        Return the points in the following order:
        1. top left
        2. top right
        3. bottom right
        4. bottom left

        Returns:
            Rectangle: Rectangle object
        """
        # pylint: disable=import-outside-toplevel
        from otary.geometry import Rectangle  # delayed import to avoid circular import

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
        from otary.geometry import Rectangle  # delayed import to avoid circular import

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
        from otary.geometry import Polygon  # delayed import to avoid circular import

        convexhull = np.squeeze(cv2.convexHull(self.asarray))
        return Polygon(convexhull)

    def distances_vertices_to_point(self, point: NDArray) -> NDArray:
        """Get the distance from all vertices in the geometry entity to the input point

        Args:
            point (NDArray): 2D point

        Returns:
            NDArray: array of the same len as the number of vertices in the geometry
                entity.
        """
        return np.linalg.norm(self.asarray - point, axis=1)

    def shortest_dist_vertices_to_point(self, point: NDArray) -> float:
        """Compute the shortest distance from the geometry entity vertices to the point

        Args:
            point (NDArray): 2D point

        Returns:
            float: shortest distance from the geometry entity vertices to the point
        """
        return np.min(self.distances_vertices_to_point(point=point))

    def longest_dist_vertices_to_point(self, point: NDArray) -> float:
        """Compute the longest distance from the geometry entity vertices to the point

        Args:
            point (NDArray): 2D point

        Returns:
            float: longest distance from the geometry entity vertices to the point
        """
        return np.max(self.distances_vertices_to_point(point=point))

    def find_vertice_ix_farthest_from(self, point: NDArray) -> int:
        """Get the index of the farthest vertice from a given point

        Args:
            point (NDArray): 2D point

        Returns:
            int: the index of the farthest vertice in the entity from the input point
        """
        return np.argmax(self.distances_vertices_to_point(point=point)).astype(int)

    def find_vertice_ix_closest_from(self, point: NDArray) -> int:
        """Get the index of the closest vertice from a given point

        Args:
            point (NDArray): 2D point

        Returns:
            int: the index of the closest point in the entity from the input point
        """
        return np.argmin(self.distances_vertices_to_point(point=point)).astype(int)

    def find_shared_approx_vertices_ix(
        self, other: DiscreteGeometryEntity, margin_dist_error: float = 5
    ) -> NDArray:
        """Compute the vertices indices from this entity that correspond to shared
        vertices with the other geometric entity.

        A vertice is considered shared if it is close enough to another vertice
        in the other geometric structure.

        Args:
            other (DiscreteGeometryEntity): other Discrete Geometry entity
            margin_dist_error (float, optional): minimum distance to have two vertices
                considered as close enough to be shared. Defaults to 5.

        Returns:
            NDArray: list of indices
        """
        return get_shared_point_indices(
            points_to_check=self.asarray,
            checkpoints=other.asarray,
            margin_dist_error=margin_dist_error,
            method="close",
            cond="any",
        )

    def find_shared_approx_vertices(
        self, other: DiscreteGeometryEntity, margin_dist_error: float = 5
    ) -> NDArray:
        """Get the shared vertices between two geometric objects.

        A vertice is considered shared if it is close enough to another vertice
        in the other geometric structure.

        Args:
            other (DiscreteGeometryEntity): a DiscreteGeometryEntity object
            margin_dist_error (float, optional): the threshold to define a vertice as
                shared or not. Defaults to 5.

        Returns:
            NDArray: list of vertices identified as shared between the two geometric
                objects
        """
        indices = self.find_shared_approx_vertices_ix(
            other=other, margin_dist_error=margin_dist_error
        )
        return self.asarray[indices]

    def find_vertices_far_from(
        self, points: NDArray, min_distance: float = 5
    ) -> NDArray:
        """Get vertices that belongs to the geometric structure far from the points in
        parameters.

        Args:
            points (NDArray): input list of points
            min_distance (float, optional): the threshold to define a point as
                far enough or not from a vertice. Defaults to 5.

        Returns:
            NDArray: vertices that belongs to the geometric structure and that
                are far from the input points.
        """
        indices = get_shared_point_indices(
            points_to_check=self.asarray,
            checkpoints=points,
            margin_dist_error=min_distance,
            method="far",
            cond="all",
        )
        return self.asarray[indices]

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, DiscreteGeometryEntity):
            return False
        if not type(self) == type(value):
            return False
        return np.array_equal(self.asarray, value.asarray)

    def __neg__(self) -> Self:
        return type(self)(-self.asarray)

    def __add__(self, other: NDArray | float | int) -> Self:
        return type(self)(self.asarray + other)

    def __sub__(self, other: NDArray | float | int) -> Self:
        return type(self)(self.asarray - other)

    def __mul__(self, other: NDArray | float | int) -> Self:
        return type(self)(self.asarray.astype(float) * other)

    def __truediv__(self, other: NDArray | float | int) -> Self:
        return type(self)(self.asarray / other)

    def __len__(self) -> int:
        return self.n_points

    def __getitem__(self, index: int) -> NDArray:
        return self.points[index]

    def __str__(self) -> str:
        return self.__class__.__name__ + "(" + self.asarray.tolist().__str__() + ")"

    def __repr__(self) -> str:
        return self.__class__.__name__ + "(" + self.asarray.tolist().__repr__() + ")"
