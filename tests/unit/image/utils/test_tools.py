"""
Unit tests for tools image funtions
"""

import pytest
import numpy as np

from otary.geometry import Point, Polygon
from otary.image.components.drawer.utils.tools import is_color_tuple, prep_obj_draw
from otary.image.utils.colors import interpolate_color
from otary.geometry.utils.tools import get_shared_point_indices


class TestToolsIsColorTuple:

    @pytest.mark.parametrize(
        "color",
        [
            (0, 0, 0),
            (255, 255, 255),
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (127, 205, 97),
        ],
    )
    def test_is_color_tuple_true(self, color):
        assert is_color_tuple(color=color)

    @pytest.mark.parametrize(
        "color",
        [
            (0, 0),
            (255, 255, 255, 255),
            (267, 0, 0),
            (0, -34, 0),
            (10000, -1, 255),
            (127, 205, "87"),
            (45, 67, (34, 45, 78)),
        ],
    )
    def test_is_color_tuple_false(self, color):
        assert not is_color_tuple(color=color)


class TestToolsInterpolateColor:

    def test_interpolate_color_red(self):
        assert interpolate_color(alpha=0) == (255, 0, 0)

    def test_interpolate_color_green(self):
        assert interpolate_color(alpha=1) == (0, 255, 0)

    def test_interpolate_color_error_alpha_neg(self):
        with pytest.raises(ValueError):
            interpolate_color(-0.001)

    def test_interpolate_color_error_alpha_toobig(self):
        with pytest.raises(ValueError):
            interpolate_color(alpha=1.001)


class TestPrepObjDraw:

    def test_prep_obj_draw_points(self):
        pt = Point([96.78, 10.67])
        objects = prep_obj_draw(objects=[pt], _type=Point)
        for element in objects:
            assert element.dtype.type is np.int32

    def test_prep_obj_draw_polygons(self):
        cnt = Polygon([[0, 0], [1, 3], [5, 0]])
        objects = prep_obj_draw(objects=[cnt], _type=Polygon)
        for element in objects:
            assert element.dtype.type is np.int32

    def test_prep_obj_draw_error_type(self):
        cnt = Polygon([[0, 0], [1, 3], [5, 0]])
        with pytest.raises(RuntimeError):
            prep_obj_draw(objects=[cnt], _type=Point)

    def test_prep_obj_draw_error_unexpected_type(self):
        with pytest.raises(RuntimeError):
            prep_obj_draw(objects=["hop", "string"], _type=str)


class TestGetSharedPointIndices:

    def test_invalid_method(self):
        points = np.array([[0, 0]])
        checkpoints = np.array([[0, 0]])
        with pytest.raises(ValueError):
            get_shared_point_indices(points, checkpoints, 1, method="invalid")

    def test_invalid_cond(self):
        points = np.array([[0, 0]])
        checkpoints = np.array([[0, 0]])
        with pytest.raises(ValueError):
            get_shared_point_indices(points, checkpoints, 1, cond="invalid")
