"""
Test Point function
"""

import numpy as np
import pytest
from otary.geometry.discrete.point import Point
from shapely.geometry import Point as SPoint


class TestPointSetArray:
    def test_asarray_getter_returns_correct_value(self):
        arr = np.array([1.0, 2.0])
        p = Point(arr)
        np.testing.assert_array_equal(p.asarray, np.array([[1.0, 2.0]]))

    def test_asarray_setter_with_flat_array(self):
        p = Point(np.array([0.0, 0.0]))
        new_val = np.array([3.0, 4.0])
        p.asarray = new_val
        print(p.asarray)
        np.testing.assert_array_equal(p.asarray, new_val.reshape((1, 2)))

    def test_asarray_setter_with_2d_array(self):
        p = Point(np.array([0.0, 0.0]))
        new_val = np.array([[5.0, 6.0]])
        p.asarray = new_val
        np.testing.assert_array_equal(p.asarray, new_val)

    def test_asarray_setter_raises_on_invalid_shape(self):
        p = Point(np.array([0.0, 0.0]))
        with pytest.raises(ValueError):
            p.asarray = np.array([[1.0, 2.0], [3.0, 4.0]])

    def test_asarray_setter_casts_to_numpy_array(self):
        p = Point(np.array([0.0, 0.0]))
        p.asarray = [7.0, 8.0]
        np.testing.assert_array_equal(p.asarray, np.array([[7.0, 8.0]]))
        

class TestPointCentroid:
    def test_centroid_returns_point_for_flat_array(self):
        arr = np.array([1.5, 2.5])
        p = Point(arr)
        np.testing.assert_array_equal(p.centroid, np.array([1.5, 2.5]))

    def test_centroid_returns_point_for_2d_array(self):
        arr = np.array([[3.0, 4.0]])
        p = Point(arr)
        np.testing.assert_array_equal(p.centroid, np.array([3.0, 4.0]))

    def test_centroid_after_asarray_setter(self):
        p = Point(np.array([0.0, 0.0]))
        p.asarray = np.array([9.0, 10.0])
        np.testing.assert_array_equal(p.centroid, np.array([9.0, 10.0]))

    def test_centroid_type_is_numpy_array(self):
        arr = np.array([5.0, 6.0])
        p = Point(arr)
        assert isinstance(p.centroid, np.ndarray)


class TestPointShapelyEdges:
    def test_shapely_edges_returns_shapely_point(self):
        arr = np.array([1.0, 2.0])
        p = Point(arr)
        shapely_point = p.shapely_edges
        # Import here to avoid test dependency on shapely if not installed
        assert isinstance(shapely_point, SPoint)
        np.testing.assert_array_equal([shapely_point.x, shapely_point.y], [1.0, 2.0])

    def test_shapely_edges_after_asarray_setter(self):
        p = Point(np.array([0.0, 0.0]))
        p.asarray = np.array([7.0, 8.0])
        shapely_point = p.shapely_edges
        assert isinstance(shapely_point, SPoint)
        np.testing.assert_array_equal([shapely_point.x, shapely_point.y], [7.0, 8.0])


class TestPointShapelySurface:
    def test_shapely_surface_returns_shapely_point(self):
        arr = np.array([2.0, 3.0])
        p = Point(arr)
        assert p.shapely_surface is None

    def test_shapely_surface_after_asarray_setter(self):
        p = Point(np.array([0.0, 0.0]))
        p.asarray = np.array([4.0, 5.0])
        assert p.shapely_surface is None


class TestPointArea:

    def test_area_returns_zero_for_flat_array(self):
        arr = np.array([1.0, 2.0])
        p = Point(arr)
        assert p.area == 0


class TestPointPerimeter:

    def test_perimeter_returns_zero_for_flat_array(self):
        arr = np.array([1.0, 2.0])
        p = Point(arr)
        assert p.perimeter == 0


class TestPointEdges:

    def test_edges_returns_empty_list_for_flat_array(self):
        arr = np.array([1.0, 2.0])
        p = Point(arr)
        assert p.edges.tolist() == []


class TestDistanceVerticesToPoint:

    def test_distance_vertices_to_point_itself(self):
        arr = np.array([1.0, 2.0])
        p = Point(arr)
        assert p.distances_vertices_to_point(point=arr) == 0

    def test_distance_vertices_to_point_other(self):
        arr = np.array([0, 0])
        other = np.array([1, 1])
        p = Point(arr)
        assert p.distances_vertices_to_point(point=other) == np.sqrt(2)


class TestOrderIdxsPointsByDist:

    def test_order_idxs_points_by_dist_desc(self):
        arr = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        assert np.all(Point.order_idxs_points_by_dist(points=arr, desc=True) == [0, 1, 2, 3])

    def test_order_idxs_points_by_dist(self):
        arr = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
        assert np.all(Point.order_idxs_points_by_dist(points=arr, desc=False) == [3, 2, 1, 0])