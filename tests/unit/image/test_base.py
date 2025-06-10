"""
Unit tests BaseImage class
"""

import numpy as np

from src.image import Image


class TestBaseImageGlobalMethods:

    def test_init_image_from_array(self):
        img = Image(image=np.full(shape=(5, 5, 3), fill_value=0))
        assert len(img.shape_array) == 3

    def test_asarray_getter(self):
        shape = (5, 5, 3)
        fill_value = 45
        img = Image.from_fillvalue(shape=shape, value=fill_value)
        assert np.array_equal(img.asarray, np.full(shape=shape, fill_value=fill_value))

    def test_asarray_setter(self):
        img = Image.from_fillvalue(shape=(5, 5, 3), value=0)
        arr = np.full(shape=(25, 25, 3), fill_value=255)
        img.asarray = arr
        assert np.array_equal(img.asarray, arr)

    def test_height(self):
        heigth = 50
        img = Image.from_fillvalue(shape=(heigth, 25), value=0)
        assert img.height == heigth

    def test_width(self):
        width = 50
        img = Image.from_fillvalue(shape=(25, width), value=0)
        assert img.width == width

    def test_area(self):
        heigth, width = 10, 7
        img = Image.from_fillvalue(shape=(heigth, width), value=0)
        assert img.area == heigth * width

    def test_center(self):
        shape = (20, 10)
        img = Image.from_fillvalue(shape=shape, value=0)
        assert np.array_equal(img.center, np.array([shape[1], shape[0]]) / 2)

    def test_norm_side_length(self):
        shape = (20, 10)
        img = Image.from_fillvalue(shape=shape, value=0)
        assert img.norm_side_length == int(np.sqrt(shape[0] * shape[1]))

    def test_asarray_norm_white(self):
        img = Image.from_fillvalue(shape=(5, 5), value=255)
        assert np.all(img.asarray_binary == 1)

    def test_asarray_norm_grey(self):
        img = Image.from_fillvalue(shape=(5, 5), value=127)
        assert np.all(np.round(img.asarray_binary, 3) == 0.498)

    def test_asarray_norm_black(self):
        img = Image.from_fillvalue(shape=(5, 5), value=0)
        assert np.all(img.asarray_binary == 0)

    def test_dist_pct(self):
        pct = 0.4
        shape = (20, 10)
        img = Image.from_fillvalue(shape=shape, value=0)
        assert img.dist_pct(pct=pct) == int(np.sqrt(shape[0] * shape[1])) * pct

    def test_corners(self):
        shape = (20, 10)
        img = Image.from_fillvalue(shape=shape, value=0)
        assert np.array_equal(img.corners[2], np.array([shape[1], shape[0]]))

    def test_copy(self):
        img = Image.from_fillvalue(shape=(5, 5), value=0)
        assert np.array_equal(img.asarray, img.copy().asarray)


class TestBaseImageIsMethods:

    def test_is_gray_true(self):
        img = Image.from_fillvalue(shape=(5, 5), value=0)
        assert img.is_gray is True

    def test_is_gray_false(self):
        img = Image.from_fillvalue(shape=(5, 5, 3), value=0)
        assert img.is_gray is False

    def test_is_shape_equal_true(self):
        shape = (5, 5, 3)
        img0 = Image.from_fillvalue(shape=shape, value=0)
        img1 = Image.from_fillvalue(shape=shape, value=127)
        assert img0.is_equal_shape(other=img1)

    def test_is_shape_equal_false(self):
        img0 = Image.from_fillvalue(shape=(5, 5, 3), value=0)
        img1 = Image.from_fillvalue(shape=(10, 7, 3), value=127)
        assert not img0.is_equal_shape(other=img1)

    def test_is_shape_equal_diff_channels_ok(self):
        img0 = Image.from_fillvalue(shape=(5, 5), value=0)  # grayscale
        img1 = Image.from_fillvalue(shape=(5, 5, 3), value=127)  # colorscale
        assert img0.is_equal_shape(other=img1, consider_channel=False)

    def test_is_shape_equal_diff_channels_notok(self):
        img0 = Image.from_fillvalue(shape=(5, 5), value=0)  # grayscale
        img1 = Image.from_fillvalue(shape=(5, 5, 3), value=127)  # colorscale
        assert not img0.is_equal_shape(other=img1)


class TestBaseImageAsMethods:

    def test_as_grayscale(self):
        img = Image.from_fillvalue(shape=(5, 5, 3), value=0).as_grayscale()
        assert len(img.shape_array) == 2

    def test_as_grayscale_nochange(self):
        img = Image.from_fillvalue(shape=(5, 5), value=0).as_grayscale()
        assert len(img.shape_array) == 2

    def test_as_colorscale(self):
        img = Image.from_fillvalue(shape=(5, 5), value=0).as_colorscale()
        assert len(img.shape_array) == 3

    def test_as_colorscale_nochange(self):
        img = Image.from_fillvalue(shape=(5, 5, 3), value=0).as_colorscale()
        assert len(img.shape_array) == 3

    def test_as_filled(self):
        val = 127
        img = Image.from_fillvalue(shape=(5, 5), value=0).as_filled(fill_value=val)
        assert np.all(img.asarray == val)

    def test_as_white(self):
        img = Image.from_fillvalue(shape=(5, 5), value=0).as_white()
        assert np.all(img.asarray == 255)

    def test_as_pil(self):
        img = Image.from_fillvalue(shape=(5, 5, 3), value=127)
        pil_img = img.as_pil()
        assert pil_img.size == (img.width, img.height)
        assert pil_img.mode == "RGB"

    def test_as_pil_grayscale(self):
        img = Image.from_fillvalue(shape=(5, 5), value=127)
        pil_img = img.as_pil()
        assert pil_img.size == (img.width, img.height)
        assert pil_img.mode == "L"


class TestTransformerImageReverseMethods:

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
