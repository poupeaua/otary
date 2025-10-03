import pytest
import numpy as np

from otary.image import Image


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


class TestImageAddBorder:
    def test_add_border_increases_shape(self):
        img = Image.from_fillvalue(shape=(10, 10), value=255)
        original_shape = img.asarray.shape
        border_size = 3
        img_with_border = img.copy().add_border(size=border_size)
        new_shape = img_with_border.asarray.shape
        assert new_shape[0] == original_shape[0] + 2 * border_size
        assert new_shape[1] == original_shape[1] + 2 * border_size

    def test_add_border_fill_value(self):
        img = Image.from_fillvalue(shape=(5, 5), value=100)
        border_size = 2
        fill_value = 42
        img_with_border = img.copy().add_border(size=border_size, fill_value=fill_value)
        # Check corners for fill_value
        assert img_with_border.asarray[0, 0] == fill_value or (
            isinstance(img_with_border.asarray[0, 0], np.ndarray)
            and np.all(img_with_border.asarray[0, 0] == fill_value)
        )
        assert img_with_border.asarray[-1, -1] == fill_value or (
            isinstance(img_with_border.asarray[-1, -1], np.ndarray)
            and np.all(img_with_border.asarray[-1, -1] == fill_value)
        )

    def test_add_border_returns_self(self):
        img = Image.from_fillvalue(shape=(5, 5), value=0)
        result = img.add_border(size=1)
        assert result is img

    def test_add_border_zero_size(self):
        img = Image.from_fillvalue(shape=(5, 5), value=0)
        img_with_border = img.copy().add_border(size=0)
        assert img_with_border.asarray.shape == img.asarray.shape

    def test_add_border_multiple_channels(self):
        img = Image.from_fillvalue(shape=(5, 5), value=128)
        border_size = 1
        fill_value = 77
        img_with_border = img.copy().add_border(size=border_size, fill_value=fill_value)
        # Check that border pixels have the correct fill value for all channels
        assert np.all(img_with_border.asarray[0, 0] == fill_value)
        assert np.all(img_with_border.asarray[-1, -1] == fill_value)
