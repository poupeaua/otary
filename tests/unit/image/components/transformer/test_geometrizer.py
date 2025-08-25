import pytest
import numpy as np

from otary.image import Image
from otary.geometry import Rectangle


class TestTransformerImageShiftMethods:

    def test_shift_xy(self):
        shape = (5, 5)
        img = Image.from_fillvalue(shape=shape, value=0)
        img.asarray[0, 0] = 255
        arr = np.full(shape=shape, fill_value=0)
        arr[4, 4] = 255
        assert np.array_equal(
            img.shift(shift=np.array([4, 4]), fill_value=0).asarray, arr
        )

    def test_shift_x(self):
        shape = (5, 5)
        img = Image.from_fillvalue(shape=shape, value=0)
        img.asarray[0, 0] = 255
        arr = np.full(shape=shape, fill_value=0)
        arr[0, 4] = 255
        assert np.array_equal(
            img.shift(shift=np.array([4, 0]), fill_value=0).asarray, arr
        )

    def test_shift_y(self):
        shape = (5, 5)
        img = Image.from_fillvalue(shape=shape, value=0)
        img.asarray[0, 0] = 255
        arr = np.full(shape=shape, fill_value=0)
        arr[4, 0] = 255
        assert np.array_equal(
            img.shift(shift=np.array([0, 4]), fill_value=0).asarray, arr
        )

    def test_shift_with_border_fill_value_tuple(self):
        img = Image.from_fillvalue(shape=(5, 5, 3), value=0)
        img.asarray[2, 2] = (255, 255, 255)
        border_fill_value = (128, 128, 128)
        img.shift(shift=np.array([2, 2]), fill_value=border_fill_value)
        assert np.all(img.asarray[0, 0] == border_fill_value)

    def test_center_to_point(self):
        val = 87
        img = Image.from_fillvalue(shape=(5, 5), value=0)
        img.asarray[0, 0] = val
        img.center_to_point(point=[0, 0])
        assert img.asarray[img.center[0], img.center[1]] == val

    def test_center_to_segment(self):
        val = 87
        img = Image.from_fillvalue(shape=(5, 5), value=0)
        img.asarray[2, 2] = val
        img.center_to_segment(segment=[[0, 0], [4, 4]])
        assert img.asarray[img.center[0], img.center[1]] == val


class TestTransformerImageRotateMethods:

    def test_rotate_360(self):
        img = Image.from_fillvalue(shape=(5, 5), value=0)
        arr = img.asarray
        img.rotate(angle=360, is_degree=True)
        assert np.array_equal(arr, img.asarray)

    def test_rotate_360_radians(self):
        img = Image.from_fillvalue(shape=(5, 5), value=0)
        arr = img.asarray
        img.rotate(angle=2 * np.pi, is_degree=False)
        assert np.array_equal(arr, img.asarray)

    @pytest.mark.parametrize(
        "angle, is_degree, is_clockwise",
        [
            (90, True, True),
            (90, True, False),
            (np.pi / 2, False, True),
            (np.pi / 2, False, False),
        ],
    )
    def test_rotate_basic(self, angle, is_degree, is_clockwise):
        img = Image.from_fillvalue(shape=(5, 5), value=0)
        img.asarray[2, 2] = 255
        img.rotate(angle=angle, is_degree=is_degree, is_clockwise=is_clockwise)
        assert img.asarray.shape == (5, 5)

    def test_rotate_with_reshape(self):
        img = Image.from_fillvalue(shape=(5, 5), value=0)
        img.asarray[2, 2] = 255
        img.rotate(angle=45, is_degree=True, reshape=True)
        assert img.asarray.shape != (5, 5)

    def test_rotate_without_reshape(self):
        img = Image.from_fillvalue(shape=(5, 5), value=0)
        img.asarray[2, 2] = 255
        img.rotate(angle=45, is_degree=True, reshape=False)
        assert img.asarray.shape == (5, 5)

    def test_rotate_check_rotate45(self):
        img = Image.from_fillvalue(shape=(100, 100, 3), value=0)
        color = (128, 245, 128)
        for i in range(10):
            img.asarray[i, :] = color
        img.rotate(angle=45, is_degree=True, fill_value=(255, 255, 255), reshape=True)

        _height_exect_color = int(img.height / 2)
        # img.draw_segments(segments=np.array([[[0, _height_exect_color], [img.width, _height_exect_color]]]))
        # img.draw_segments(segments=np.array([[[_height_exect_color, 0], [_height_exect_color, img.height]]]))

        assert np.array_equal(img.asarray[_height_exect_color, -10], color)
        assert np.array_equal(img.asarray[10, _height_exect_color], color)

    def test_rotate_check_rotate70(self):
        img = Image.from_fillvalue(shape=(100, 100, 3), value=0)
        color = (128, 245, 128)
        for i in range(10):
            img.asarray[i, :] = color

        img.rotate(angle=70, is_degree=True, fill_value=(255, 255, 255), reshape=True)

        _height_exect_color = int(img.height * 0.71)
        # img.draw_segments(segments=np.array([[[0, _height_exect_color], [img.width, _height_exect_color]]]))
        # img.draw_segments(segments=np.array([[[_height_exect_color, 0], [_height_exect_color, img.height]]]))

        assert np.array_equal(img.asarray[_height_exect_color, -10], color)
        assert np.array_equal(img.asarray[10, _height_exect_color], color)

    def test_rotate_negative_angle(self):
        img = Image.from_fillvalue(shape=(5, 5), value=0)
        img.asarray[2, 2] = 255
        img.rotate(angle=-90, is_degree=True)
        assert img.asarray.shape == (5, 5)

    def test_rotate_with_border_fill_value_tuple(self):
        img = Image.from_fillvalue(shape=(5, 5, 3), value=0)
        border_fill_value = (128, 128, 128)
        img.rotate(angle=45, is_degree=True, fill_value=border_fill_value, reshape=True)
        assert np.all(img.asarray[0, 0] == border_fill_value)

    def test_rotate_without_reshape_with_border_fill_value_tuple(self):
        img = Image.from_fillvalue(shape=(5, 5, 3), value=0)
        border_fill_value = (255, 0, 0)
        img.rotate(
            angle=90, is_degree=True, fill_value=border_fill_value, reshape=False
        )
        assert np.all(img.asarray[0, 0] == border_fill_value)

    def test_rotate_negative_angle_with_border_fill_value_tuple(self):
        img = Image.from_fillvalue(shape=(5, 5, 3), value=0)
        img.asarray[2, 2] = (255, 255, 255)
        border_fill_value = (0, 255, 0)
        img.rotate(angle=-45, is_degree=True, fill_value=border_fill_value)
        assert np.all(img.asarray[0, 0] == border_fill_value)


class TestImageRestrictRectInFrame:

    def test_restrict_rect_in_frame_within_bounds(self):
        img = Image.from_fillvalue(shape=(10, 10), value=255)
        rect = Rectangle.from_topleft_bottomright(
            topleft=np.asarray([2, 2]),
            bottomright=np.asarray([8, 8]),
            is_cast_int=True,
        )
        restricted_rect = img.restrict_rect_in_frame(rect)
        assert restricted_rect.xmin == 2
        assert restricted_rect.ymin == 2
        assert restricted_rect.xmax == 8
        assert restricted_rect.ymax == 8

    def test_restrict_rect_in_frame_outside_bounds(self):
        img = Image.from_fillvalue(shape=(10, 10), value=255)
        rect = Rectangle.from_topleft_bottomright(
            topleft=np.asarray([-5, -5]),
            bottomright=np.asarray([15, 15]),
            is_cast_int=True,
        )
        restricted_rect = img.restrict_rect_in_frame(rect)
        assert restricted_rect.xmin == 0
        assert restricted_rect.ymin == 0
        assert restricted_rect.xmax == 10
        assert restricted_rect.ymax == 10

    def test_restrict_rect_in_frame_partial_outside_bounds(self):
        img = Image.from_fillvalue(shape=(10, 10), value=255)
        rect = Rectangle.from_topleft_bottomright(
            topleft=np.asarray([5, 5]),
            bottomright=np.asarray([15, 15]),
            is_cast_int=True,
        )
        restricted_rect = img.restrict_rect_in_frame(rect)
        assert restricted_rect.xmin == 5
        assert restricted_rect.ymin == 5
        assert restricted_rect.xmax == 10
        assert restricted_rect.ymax == 10

    def test_restrict_rect_in_frame_exact_bounds(self):
        img = Image.from_fillvalue(shape=(10, 10), value=255)
        rect = Rectangle.from_topleft_bottomright(
            topleft=np.asarray([0, 0]),
            bottomright=np.asarray([10, 10]),
            is_cast_int=True,
        )
        restricted_rect = img.restrict_rect_in_frame(rect)
        assert restricted_rect.xmin == 0
        assert restricted_rect.ymin == 0
        assert restricted_rect.xmax == 10
        assert restricted_rect.ymax == 10
