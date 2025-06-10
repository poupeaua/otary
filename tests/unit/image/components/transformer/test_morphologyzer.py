import pytest
import numpy as np

from src.image import Image


class TestTransformerImageResizeMethods:

    def test_resize_fixed(self):
        img = Image.from_fillvalue(shape=(5, 5), value=0)
        new_shape = (10, 15)
        img.resize_fixed(dim=new_shape)
        assert img.shape_array == (new_shape[1], new_shape[0])

    def test_resize_fixed_height_neg(self):
        img = Image.from_fillvalue(shape=(5, 7), value=0)
        new_shape = (10, -1)
        img.resize_fixed(dim=new_shape)
        assert img.shape_array == (7, new_shape[0])

    def test_resize_fixed_width_neg(self):
        img = Image.from_fillvalue(shape=(10, 20), value=0)
        new_shape = (-1, 20)
        img.resize_fixed(dim=new_shape)
        assert img.shape_array == (new_shape[1], 40)

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


class TestTransformerMorphologyBlur:

    def test_blur_basic(self):
        img = Image.from_fillvalue(shape=(5, 5), value=0)
        img.asarray[2, 2] = 255
        img.blur()
        assert img.asarray[2, 1] > 0

    def test_blur_gaussian(self):
        img = Image.from_fillvalue(shape=(5, 5), value=0)
        img.asarray[2, 2] = 255
        img.blur(method="gaussian")
        assert img.asarray[2, 1] > 0

    def test_blur_median(self):
        img = Image.from_fillvalue(shape=(5, 5), value=0)
        img.asarray[1, 1] = 255
        img.asarray[1, 2] = 255
        img.asarray[1, 3] = 255
        img.asarray[2, 1] = 255
        img.asarray[2, 3] = 255
        img.blur(method="median", kernel=(3, 3))
        assert img.asarray[2, 2] > 0

    def test_blur_bilateral(self):
        img = Image.from_fillvalue(shape=(5, 5), value=0)
        img.asarray[1, 1] = 255
        img.asarray[1, 2] = 255
        img.asarray[1, 3] = 255
        img.asarray[2, 1] = 255
        img.asarray[2, 3] = 255
        img.blur(method="bilateral", kernel=(3, 3))
        assert img.asarray[2, 2] > 0

    def test_blur_invalid_method(self):
        img = Image.from_fillvalue(shape=(5, 5), value=0)
        img.asarray[2, 2] = 255
        with pytest.raises(ValueError):
            img.blur(method="not_a_valid_method")


class TestTransformerMorphologyDilate:

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

    def test_dilate_iterations_zero(self):
        arr = np.array(
            [
                [255, 127, 0, 255, 224],
                [0, 127, 255, 255, 0],
                [255, 230, 127, 0, 255],
                [0, 0, 0, 127, 179],
                [0, 127, 0, 255, 198],
            ]
        )
        img = Image(arr)
        img.dilate(iterations=0)
        assert np.all(img.asarray == arr)


class TestTransformerMorphologyErode:

    def test_erode_black(self):
        img = Image.from_fillvalue(shape=(5, 5), value=255)
        img.asarray[2, 2] = 0
        img.erode(kernel=(3, 3), iterations=2)
        assert np.all(img.asarray == 255)

    def test_dilate_hard_case(self):
        arr = np.array(
            [
                [255, 127, 0, 255, 224],
                [0, 127, 255, 255, 0],
                [255, 230, 127, 0, 255],
                [0, 0, 0, 127, 179],
                [0, 127, 0, 255, 198],
            ]
        )
        img = Image(arr)
        img.dilate(iterations=3)
        assert np.all(img.asarray == 0)

    def test_erode_white(self):
        img = Image.from_fillvalue(shape=(5, 5), value=255)
        img.asarray[2, 2] = 0
        img.erode(kernel=(3, 3), iterations=2, erode_black_pixels=False)
        assert np.all(img.asarray == 0)

    def test_erode_iterations_zero(self):
        arr = np.array(
            [
                [255, 127, 0, 255, 224],
                [0, 127, 255, 255, 0],
                [255, 230, 127, 0, 255],
                [0, 0, 0, 127, 179],
                [0, 127, 0, 255, 198],
            ]
        )
        img = Image(arr)
        img.erode(iterations=0)
        assert np.all(img.asarray == arr)

    def test_erode_hard_case(self):
        arr = np.array(
            [
                [255, 127, 0, 255, 224],
                [0, 127, 255, 255, 0],
                [255, 230, 127, 0, 255],
                [0, 0, 0, 127, 179],
                [0, 127, 0, 255, 198],
            ]
        )
        img = Image(arr)
        img.erode(iterations=3)
        assert np.all(img.asarray == 255)
