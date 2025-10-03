import pytest
import numpy as np

from otary.image import Image


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