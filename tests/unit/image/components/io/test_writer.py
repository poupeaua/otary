"""
Unit tests for the writer image class
"""

from unittest import mock
import numpy as np
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
