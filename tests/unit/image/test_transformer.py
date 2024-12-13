"""
Unit Tests for the transformer image methods
"""

import pytest
import numpy as np

from src.image import Image


class TestTransformerImageThresholdMethods:

    @pytest.mark.parametrize("thresh", [25, 50, 100, 150])
    def test_threshold_simple(self, thresh):
        img = Image.from_fillvalue(shape=(5, 5), value=thresh - 1)
        img.asarray[0, 0] = 255
        img.threshold_simple(thresh=thresh)
        assert img.asarray[0, 0] == 255
        img.asarray[0, 0] = 0
        assert np.all(img.asarray == 0)

    def test_threshold_otsu(self):
        img = Image.from_fillvalue(shape=(5, 5), value=45)
        img.threshold_otsu()
        assert np.all(img.asarray == 0)

    def test_threshold_sauvola(self):
        img = Image.from_fillvalue(shape=(5, 5), value=127)
        img.threshold_sauvola()
        assert np.all(img.asarray == 255)


class TestTransformerImageBinary:

    def test_binary_otsu(self):
        img = Image.from_fillvalue(shape=(5, 5), value=0)
        assert np.all(img.binary(method="otsu") == 0)

    def test_binary_sauvola(self):
        img = Image.from_fillvalue(shape=(5, 5), value=55)
        assert np.all(img.binary(method="sauvola") == 1)

    def test_binaryrev_otsu(self):
        img = Image.from_fillvalue(shape=(5, 5), value=0)
        assert np.all(img.binaryrev(method="otsu") == 1)

    def test_binaryrev_sauvola(self):
        img = Image.from_fillvalue(shape=(5, 5), value=55)
        assert np.all(img.binaryrev(method="sauvola") == 0)

    def test_binary_error_method(self):
        img = Image.from_fillvalue(shape=(5, 5), value=55)
        with pytest.raises(ValueError):
            img.binary(method="not_an_expected_binary_method")


class TestTransformerImageGlobalMethods:

    def test_rev_grayscale_black(self):
        img = Image.from_fillvalue(shape=(5, 5), value=0)
        img.rev()
        assert np.all(img.asarray == 255)

    def test_rev_grayscale_white(self):
        img = Image.from_fillvalue(shape=(5, 5), value=255)
        img.rev()
        assert np.all(img.asarray == 0)

    def test_rev_grayscale_other_color(self):
        val = 100
        img = Image.from_fillvalue(shape=(5, 5), value=val)
        img.rev()
        assert np.all(img.asarray == np.abs(255 - val))

    def test_rev_colorscale_black(self):
        img = Image.from_fillvalue(shape=(5, 5, 3), value=0)
        img.rev()
        assert np.all(img.asarray == 255)

    def test_rev_colorscale_white(self):
        img = Image.from_fillvalue(shape=(5, 5, 3), value=255)
        img.rev()
        assert np.all(img.asarray == 0)

    def test_rev_colorscale_other_color(self):
        val = 100
        img = Image.from_fillvalue(shape=(5, 5, 3), value=val)
        img.rev()
        assert np.all(img.asarray == 155)

    def test_blur(self):
        img = Image.from_fillvalue(shape=(5, 5), value=0)
        img.asarray[2, 2] = 255
        img.blur()
        assert img.asarray[2, 1] > 0

    def test_dilate_black(self):
        img = Image.from_fillvalue(shape=(5, 5), value=255)
        img.asarray[2, 2] = 0
        img.dilate(kernel=(3, 3), iterations=2)
        assert np.all(img.asarray == 0)

    def test_dilate_white(self):
        img = Image.from_fillvalue(shape=(5, 5), value=0)
        img.asarray[2, 2] = 255
        img.dilate(kernel=(3, 3), iterations=2, dilate_black_pixels=False)
        assert np.all(img.asarray == 255)

    def test_shift(self):
        shape = (5, 5)
        img = Image.from_fillvalue(shape=shape, value=0)
        img.asarray[0, 0] = 255
        arr = np.full(shape=shape, fill_value=0)
        arr[4, 4] = 255
        assert np.array_equal(img.shift(shift=np.array([4, 4])).asarray, arr)

    def test_rotate_180(self):
        img = Image.from_fillvalue(shape=(5, 5), value=0)
        img.asarray[1, 2] = 1
        img.asarray[1, 4] = 1
        img.rotate(angle=180, is_degree=True)
        assert img.asarray[3, 0] == 1
        assert img.asarray[1, 4] == 0
        assert img.asarray[3, 2] == 1
        assert img.asarray[1, 2] == 0

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

    def test_center_image_to_point(self):
        val = 87
        img = Image.from_fillvalue(shape=(5, 5), value=0)
        img.asarray[0, 0] = val
        img.center_image_to_point(point=[0, 0])
        assert img.asarray[*img.center] == val

    def test_center_image_to_segment(self):
        val = 87
        img = Image.from_fillvalue(shape=(5, 5), value=0)
        img.asarray[2, 2] = val
        img.center_image_to_segment(segment=[[0, 0], [4, 4]])
        assert img.asarray[*img.center] == val


class TestTransformerImageResizeMethods:

    def test_resize_fixed(self):
        img = Image.from_fillvalue(shape=(5, 5), value=0)
        new_shape = (10, 15)
        img.resize_fixed(dim=new_shape)
        assert img.shape_array == new_shape

    def test_resize_fixed_height_neg(self):
        img = Image.from_fillvalue(shape=(5, 7), value=0)
        new_shape = (10, -1)
        img.resize_fixed(dim=new_shape)
        assert img.shape_array == (10, 14)

    def test_resize_fixed_width_neg(self):
        img = Image.from_fillvalue(shape=(10, 20), value=0)
        new_shape = (-1, 10)
        img.resize_fixed(dim=new_shape)
        assert img.shape_array == (5, 10)

    def test_resize_fixed_error(self):
        img = Image.from_fillvalue(shape=(5, 5), value=0)
        new_shape = (-1, -23)
        with pytest.raises(ValueError):
            img.resize_fixed(dim=new_shape)

    @pytest.mark.parametrize("factor", [0.5, 1, 2, 3])
    def test_resize(self, factor):
        init_shape = (10, 10)
        img = Image.from_fillvalue(shape=init_shape, value=0)
        img.resize(factor=factor)
        assert img.shape_array == tuple(np.array(init_shape) * factor)

    def test_resize_factor_neg(self):
        img = Image.from_fillvalue(shape=(5, 5), value=0)
        with pytest.raises(ValueError):
            img.resize(factor=-2)

    def test_resize_factor_toobig(self):
        img = Image.from_fillvalue(shape=(5, 5), value=0)
        with pytest.raises(ValueError):
            img.resize(factor=100)


class TestTransformerImageCropMethods:

    def test_crop(self):
        img = Image.from_fillvalue(shape=(5, 5), value=0)
        img.crop(1, 1, 4, 3)
        assert img.shape_array == (3, 4)

    def test_crop_around_segment_horizontal(self):
        img = Image.from_fillvalue(shape=(5, 5), value=0)
        img, _, _, _ = img.crop_around_segment_horizontal(
            segment=[[1, 2], [3, 2]], dim_crop_rect=(-1, 2), default_extra_width=0
        )
        assert img.shape_array == (3, 3)
