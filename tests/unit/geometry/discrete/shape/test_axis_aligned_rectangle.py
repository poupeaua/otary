"""
Tests for the AxisAlignedRectangle class
"""

import pymupdf
import pytest
import numpy as np

from otary.geometry import AxisAlignedRectangle, Rectangle
from otary.geometry.discrete.shape.polygon import Polygon


class TestAxisAlignedRectangleCreation:

    def test_create_axis_aligned_rectangle_base(self):
        # Create an axis-aligned rectangle
        rect = AxisAlignedRectangle.from_topleft(topleft=[0, 0], width=2, height=4)

        # Assert the rectangle has correct coordinates
        assert rect.xmin == 0
        assert rect.ymin == 0
        assert rect.xmax == 2
        assert rect.ymax == 4

        # assert first point is top-left and points are ordered clockwise
        assert (rect.points[0] == [0, 0]).all()
        assert (rect.points[1] == [2, 0]).all()
        assert (rect.points[2] == [2, 4]).all()
        assert (rect.points[3] == [0, 4]).all()

    def test_create_axis_aligned_rectangle_from_center(self):
        # Create an axis-aligned rectangle from center
        rect = AxisAlignedRectangle.from_center(center=[5, 5], width=4, height=2)

        # Assert the rectangle has correct coordinates
        assert rect.xmin == 3
        assert rect.ymin == 4
        assert rect.xmax == 7
        assert rect.ymax == 6

        # assert first point is top-left and points are ordered clockwise
        assert (rect.points[0] == [3, 4]).all()
        assert (rect.points[1] == [7, 4]).all()
        assert (rect.points[2] == [7, 6]).all()
        assert (rect.points[3] == [3, 6]).all()

    def test_create_axis_aligned_rectangle_from_rectangle(self):
        # Create an axis-aligned rectangle from a Rectangle object
        rect = Rectangle.from_center(center=[5, 5], width=4, height=2)
        rect = AxisAlignedRectangle.from_rectangle(rectangle=rect)

        # Assert the rectangle has correct coordinates
        assert rect.xmin == 3
        assert rect.ymin == 4
        assert rect.xmax == 7
        assert rect.ymax == 6

    def test_create_non_axis_aligned_rectangle_raises(self):
        # Attempt to create a non-axis-aligned rectangle
        with pytest.raises(ValueError):
            rect = Rectangle.from_center(center=[2, 5], width=6, height=10).rotate(
                angle=30
            )
            AxisAlignedRectangle.from_rectangle(rectangle=rect)


class TestAxisAlignedRectangleProperties:

    def test_height_and_width_properties(self):
        # Create an axis-aligned rectangle
        rect = AxisAlignedRectangle.from_topleft(topleft=[0, 0], width=3, height=5)

        # Assert the height and width properties are correct
        assert rect.height == 5
        assert rect.width == 3

    def test_area_property(self):
        # Create an axis-aligned rectangle
        width = 3
        height = 5
        rect = AxisAlignedRectangle.from_topleft(topleft=[0, 0], width=3, height=5)

        # Assert the area property is correct
        assert rect.area == width * height

    def test_perimeter_property(self):
        # Create an axis-aligned rectangle
        width = 3
        height = 5
        rect = AxisAlignedRectangle.from_topleft(
            topleft=[0, 0], width=width, height=height
        )

        # Assert the perimeter property is correct
        assert rect.perimeter == 2 * (width + height)


class TestAxisAlignedRectanglePyMuRect:

    def test_as_pymupdf_rect_base(self):
        # Create an axis-aligned rectangle
        rect = AxisAlignedRectangle.from_topleft(topleft=[0, 0], width=2, height=4)
        pymupdf_rect = rect.as_pymupdf_rect

        # Assert the pymupdf.Rect object has correct coordinates
        assert isinstance(pymupdf_rect, pymupdf.Rect)
        assert pymupdf_rect.x0 == 0
        assert pymupdf_rect.y0 == 0
        assert pymupdf_rect.x1 == 2
        assert pymupdf_rect.y1 == 4


class TestAxisAlignedRectangleShift:

    def test_shift_base(self):
        # Create an axis-aligned rectangle
        rect = AxisAlignedRectangle.from_topleft(topleft=[1, 1], width=3, height=2)

        # Shift the rectangle by (2, 3)
        rect.shift(vector=[2, 3])

        # Assert the new coordinates are correct
        assert rect.xmin == 3
        assert rect.ymin == 4
        assert rect.xmax == 6
        assert rect.ymax == 6


class TestAxisAlignedRectangleRotated90:

    def test_rotated90_assert_axis_aligned(self):
        # Create an axis-aligned rectangle
        rect = AxisAlignedRectangle.from_topleft(topleft=[1, 1], width=3, height=2)

        # Rotate the rectangle by 90 degrees
        rect_rot = rect.rotated90
        assert rect_rot.is_axis_aligned

    def test_rotated90_assert_new_coords(self):
        # Create an axis-aligned rectangle
        rect = AxisAlignedRectangle.from_topleft(topleft=[1, 1], width=3, height=2)
        rect_rot = rect.rotated90

        # Assert the new coordinates are correct
        assert rect_rot.width == rect.height
        assert rect_rot.height == rect.width
        assert rect_rot.centroid[0] == rect.centroid[0]
        assert rect_rot.centroid[1] == rect.centroid[1]

    def test_rotated90_assert_topleft_and_clockwise(self):
        # Create an axis-aligned rectangle
        rect = AxisAlignedRectangle.from_topleft(topleft=[1, 1], width=3, height=2)
        rect_rot = rect.rotated90

        # assert first point is top-left and points are ordered clockwise
        assert (rect_rot.asarray[0] == [1.5, 0.5]).all()
        assert rect_rot.is_clockwise(is_y_axis_down=True)


class TestAxisAlignedRectangleRotateTransform:

    def test_rotate_transform_base(self):
        # Create an axis-aligned rectangle
        rect = AxisAlignedRectangle.from_topleft(topleft=[1, 1], width=3, height=2)

        # Rotate the rectangle by 45 degrees
        rect_rot = rect.rotate_transform(angle=45)

        assert not rect_rot.is_axis_aligned
        assert not isinstance(rect_rot, AxisAlignedRectangle)


class TestAxisAlignedRectangleRotate:

    def test_rotate_raises(self):
        # Create an axis-aligned rectangle
        rect = AxisAlignedRectangle.from_topleft(topleft=[1, 1], width=3, height=2)

        # Attempt to rotate the rectangle by 45 degrees
        with pytest.raises(TypeError):
            rect.rotate(angle=45)


class TestAxisAlignedRectangleCopy:

    def test_copy_base(self):
        # Create an axis-aligned rectangle
        rect = AxisAlignedRectangle.from_topleft(topleft=[1, 1], width=3, height=2)

        # Create a copy of the rectangle
        rect_copy = rect.copy()

        assert isinstance(rect_copy, AxisAlignedRectangle)

        # Assert the copy has the same properties as the original
        assert rect_copy.xmin == rect.xmin
        assert rect_copy.ymin == rect.ymin
        assert rect_copy.xmax == rect.xmax
        assert rect_copy.ymax == rect.ymax
        assert (rect_copy.asarray == rect.asarray).all()

        # Assert the copy is a different object
        assert rect_copy is not rect


class TestAxisAlignedRectangleFromPolygon:

    def test_from_polygon_with_axis_aligned_polygon(self):
        # create an axis aligned rectangle and use its point array as polygon
        rect = AxisAlignedRectangle.from_topleft(topleft=[2, 3], width=4, height=6)
        polygon = rect.asarray

        rect_from_poly = AxisAlignedRectangle.from_polygon(polygon=polygon)

        assert isinstance(rect_from_poly, AxisAlignedRectangle)
        assert rect_from_poly.xmin == rect.xmin
        assert rect_from_poly.ymin == rect.ymin
        assert rect_from_poly.xmax == rect.xmax
        assert rect_from_poly.ymax == rect.ymax

    def test_axis_aligned_rectangle_input(self):
        """A rectangle whose AABB is itself should round-trip exactly."""
        pts = np.array([[0, 0], [4.5, -1], [4, 3], [0, 3.3]], dtype=np.float32)
        polygon = Polygon(pts)
        rect = AxisAlignedRectangle.from_polygon(polygon)
        assert rect.xmin == pytest.approx(0.0)
        assert rect.xmax == pytest.approx(4.5)
        assert rect.ymin == pytest.approx(-1)
        assert rect.ymax == pytest.approx(3.3)

    def test_axis_aligned_rectangle_input(self):
        """A rectangle whose AABB is itself should round-trip exactly."""
        pts = np.array([[0, 0], [4, 0], [4, 3], [0, 3]], dtype=np.float32)
        rect = AxisAlignedRectangle.from_polygon(pts)
        assert rect.xmin == pytest.approx(0.0)
        assert rect.xmax == pytest.approx(4.0)
        assert rect.ymin == pytest.approx(0.0)
        assert rect.ymax == pytest.approx(3.0)

    def test_triangle(self):
        pts = np.array([[1, 2], [5, 0], [3, 6]], dtype=np.float32)
        rect = AxisAlignedRectangle.from_polygon(pts)
        assert rect.xmin == pytest.approx(1.0)
        assert rect.xmax == pytest.approx(5.0)
        assert rect.ymin == pytest.approx(0.0)
        assert rect.ymax == pytest.approx(6.0)

    def test_negative_coordinates(self):
        pts = np.array([[-3, -7], [-1, -2], [-5, -1]], dtype=np.float32)
        rect = AxisAlignedRectangle.from_polygon(pts)
        assert rect.xmin == pytest.approx(-5.0)
        assert rect.xmax == pytest.approx(-1.0)
        assert rect.ymin == pytest.approx(-7.0)
        assert rect.ymax == pytest.approx(-1.0)

    def test_mixed_sign_coordinates(self):
        pts = np.array([[-10, 5], [10, -5], [0, 0]], dtype=np.float32)
        rect = AxisAlignedRectangle.from_polygon(pts)
        assert rect.xmin == pytest.approx(-10.0)
        assert rect.xmax == pytest.approx(10.0)
        assert rect.ymin == pytest.approx(-5.0)
        assert rect.ymax == pytest.approx(5.0)

    def test_self_intersected_aabb(self):
        pts = np.array([[0, 0], [100, 100], [0, 100], [100, 0]], dtype=np.float32)
        rect = AxisAlignedRectangle.from_polygon(pts)
        assert rect.xmin == pytest.approx(0.0)
        assert rect.xmax == pytest.approx(100.0)
        assert rect.ymin == pytest.approx(0.0)
        assert rect.ymax == pytest.approx(100.0)


class TestAxisAlignedRectangleFromPolygonErrors:

    def test_error_empty_polygon(self):
        pts = "nonsense"
        with pytest.raises(TypeError):
            AxisAlignedRectangle.from_polygon(pts)

    def test_error_single_point_degenerates_to_zero_area(self):
        pts = np.array([[3, 7]], dtype=np.float32)
        with pytest.raises(ValueError):
            AxisAlignedRectangle.from_polygon(pts)

    def test_collinear_points_error_self_intersected(self):
        pts = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=np.float32)
        with pytest.raises(ValueError):
            AxisAlignedRectangle.from_polygon(pts)

    def test_1d_array_raises_value_error(self):
        with pytest.raises(ValueError):
            AxisAlignedRectangle.from_polygon(np.array([0, 1, 2, 3]))

    def test_3d_points_raises_value_error(self):
        with pytest.raises(ValueError):
            AxisAlignedRectangle.from_polygon(np.array([[0, 1, 2], [3, 4, 5]]))

    def test_wrong_second_dim_raises_value_error(self):
        with pytest.raises(ValueError):
            AxisAlignedRectangle.from_polygon(np.ones((5, 3)))


# ---------------------------------------------------------------------------
# Random-polygon tests (fixed seed for reproducibility)
# ---------------------------------------------------------------------------


def make_random_polygon_array(n_points: int, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-1000, 1000, size=(n_points, 2)).astype(np.float32)


class TestFromPolygonRandom:

    @pytest.mark.parametrize("seed", range(20))
    def test_bounding_box_contains_all_points(self, seed: int):
        """Every input point must lie inside (or on the edge of) the AABB."""
        pts = make_random_polygon_array(n_points=50, seed=seed)
        rect = AxisAlignedRectangle.from_polygon(pts)
        assert np.all(pts[:, 0] >= rect.xmin - 1e-5)
        assert np.all(pts[:, 0] <= rect.xmax + 1e-5)
        assert np.all(pts[:, 1] >= rect.ymin - 1e-5)
        assert np.all(pts[:, 1] <= rect.ymax + 1e-5)

    @pytest.mark.parametrize("seed", range(20))
    def test_bounding_box_is_tight(self, seed: int):
        """At least one point must touch each of the four sides."""
        pts = make_random_polygon_array(n_points=50, seed=seed)
        rect = AxisAlignedRectangle.from_polygon(pts)
        tol = 1e-5
        assert np.any(pts[:, 0] <= rect.xmin + tol), "No point on left edge"
        assert np.any(pts[:, 0] >= rect.xmax - tol), "No point on right edge"
        assert np.any(pts[:, 1] <= rect.ymin + tol), "No point on bottom edge"
        assert np.any(pts[:, 1] >= rect.ymax - tol), "No point on top edge"

    @pytest.mark.parametrize("seed", range(20))
    def test_output_has_four_corners(self, seed: int):
        pts = make_random_polygon_array(n_points=30, seed=seed)
        rect = AxisAlignedRectangle.from_polygon(pts)
        assert len(rect.asarray) == 4

    @pytest.mark.parametrize("n_points", [3, 10, 100, 500])
    def test_various_polygon_sizes(self, n_points: int):
        pts = make_random_polygon_array(n_points=n_points, seed=42)
        rect = AxisAlignedRectangle.from_polygon(pts)
        assert rect.xmin <= rect.xmax
        assert rect.ymin <= rect.ymax

    def test_float32_input_preserves_dtype(self):
        pts = make_random_polygon_array(50, seed=7).astype(np.float32)
        rect = AxisAlignedRectangle.from_polygon(pts)
        # Bounds should still be numerically correct despite float32 input
        assert rect.xmin == pytest.approx(float(pts[:, 0].min()), rel=1e-5)
        assert rect.xmax == pytest.approx(float(pts[:, 0].max()), rel=1e-5)

    def test_polygon_object_matches_ndarray(self):
        """Passing a Polygon object and the equivalent ndarray must yield the same AABB."""
        pts = make_random_polygon_array(40, seed=99)
        polygon_obj = Polygon(pts)
        rect_from_obj = AxisAlignedRectangle.from_polygon(polygon_obj)
        rect_from_arr = AxisAlignedRectangle.from_polygon(pts)
        assert rect_from_obj.xmin == pytest.approx(rect_from_arr.xmin)
        assert rect_from_obj.xmax == pytest.approx(rect_from_arr.xmax)
        assert rect_from_obj.ymin == pytest.approx(rect_from_arr.ymin)
        assert rect_from_obj.ymax == pytest.approx(rect_from_arr.ymax)
