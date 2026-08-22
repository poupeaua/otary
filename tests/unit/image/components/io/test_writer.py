"""
Unit tests for the writer image class
"""

from unittest import mock
import numpy as np
import pytest
from otary.image import Image


class TestWriterShow:

    def test_show_base(self):
        arr = np.ones((10, 10, 3), dtype=np.uint8)
        im = Image(arr)
        im.show(figsize=(50, 50))

    def test_show_no_params(self):
        arr = np.ones((3, 3, 3), dtype=np.uint8)
        im = Image(arr)
        im.show()

    @pytest.mark.parametrize(
        "dtype",
        [
            np.float32,
            np.float64,
            np.int32,
            np.uint16,
        ],
    )
    def test_show_non_uint8_image(self, dtype):
        arr = np.ones((3, 3, 3), dtype=dtype)
        im = Image(arr)
        im.show()

    @mock.patch("otary.image.components.io.writer.ImagePIL.fromarray")
    def test_show_clips_non_uint8_image(self, mock_fromarray):
        arr = np.array([[[-10, 0, 300]]], dtype=np.float32)
        im = Image(arr)
        im.show()

        passed_array = mock_fromarray.call_args.args[0]

        assert passed_array.dtype == np.uint8
        assert passed_array.min() == 0
        assert passed_array.max() == 255

    @mock.patch("PIL.Image.Image.show")
    def test_show_popup_window(self, mock_show):
        arr = np.ones((3, 3, 3), dtype=np.uint8)
        im = Image(arr)
        im.show(popup_window_display=True)


class TestWriterSave:

    @mock.patch("PIL.Image.Image.save")
    def test_save_calls_show_with_filepath(self, mock_save):
        arr = np.ones((100, 100, 3), dtype=np.uint8)
        im = Image(arr)
        im.save("output.png")

    @mock.patch("PIL.Image.Image.save")
    def test_save_with_different_filepath(self, mock_save):
        arr = np.ones((100, 100, 3), dtype=np.uint8) * 226
        im = Image(arr)
        im.save("another_file.jpg")
