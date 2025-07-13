import cv2
import pytest
from unittest import mock
import numpy as np
from otary.image import Image

"""
Unit tests for the writer image class
"""

class TestWriterShow:
    
    @mock.patch("otary.image.components.io.writer.plt")
    def test_show_with_all_args(self, mock_plt):
        arr = np.ones((10, 10, 3), dtype=np.uint8)
        im = Image(arr)

        im.show(
            title="Test Image",
            figsize=(5, 5),
            save_filepath="test.png"
        )

        mock_plt.figure.assert_called_once_with(figsize=(5, 5))
        mock_plt.xticks.assert_called_once_with([])
        mock_plt.yticks.assert_called_once_with([])
        mock_plt.title.assert_called_once_with("Test Image")
        mock_plt.savefig.assert_called_once_with("test.png")
        mock_plt.show.assert_called_once()

    @mock.patch("otary.image.components.io.writer.plt")
    def test_show_without_color_conversion(self, mock_plt):
        arr = np.ones((5, 5, 3), dtype=np.uint8)
        im = Image(arr)

        im.show(color_conversion=None)

        mock_plt.imshow.assert_called_once_with(im.asarray)

    @mock.patch("otary.image.components.io.writer.plt")
    def test_show_without_title_and_save(self, mock_plt):
        arr = np.ones((3, 3, 3), dtype=np.uint8)
        im = Image(arr)

        im.show()

        mock_plt.title.assert_not_called()
        mock_plt.savefig.assert_not_called()
        mock_plt.show.assert_called_once()

 
class TestWriterSave:

    @mock.patch("otary.image.components.io.writer.WriterImage.show")
    def test_save_calls_show_with_filepath(self, mock_show):
        arr = np.ones((4, 4, 3), dtype=np.uint8)
        im = Image(arr)
        im.save("output.png")
        mock_show.assert_called_once_with(save_filepath="output.png")

    @mock.patch("otary.image.components.io.writer.WriterImage.show")
    def test_save_with_different_filepath(self, mock_show):
        arr = np.ones((2, 2, 3), dtype=np.uint8)
        im = Image(arr)
        im.save("another_file.jpg")
        mock_show.assert_called_once_with(save_filepath="another_file.jpg")
