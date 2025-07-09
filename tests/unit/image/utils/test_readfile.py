"""
Unit test to read file relative to image processing
"""

import io
import pytest
import numpy as np

from otary.image.components.io.utils.readfile import read_pdf_to_images


@pytest.fixture
def pdf_filepath():
    return "./tests/data/test.pdf"


@pytest.fixture
def txt_filepath():
    return "./tests/data/test.txt"


class TestReadPdfFile:

    @pytest.mark.parametrize("res", [100, 200, 300])
    def test_read_pdf_file(self, pdf_filepath, res):
        images = read_pdf_to_images(filepath_or_stream=pdf_filepath, resolution=res)
        assert len(images) == 1  # ensure we have only one pdf page read
        assert type(images[0]) == np.ndarray  # ensure output type is numpy array
        assert images[0].shape[0] == res  # assert resolution

    def test_read_pdf_file_resolution_none(self, pdf_filepath):
        images = read_pdf_to_images(filepath_or_stream=pdf_filepath, resolution=None)
        assert len(images) == 1
        assert type(images[0]) == np.ndarray

    def test_read_pdf_stream(self, pdf_filepath):
        with open(pdf_filepath, "rb") as pdf_file:
            pdf_stream = io.BytesIO(pdf_file.read())
        images = read_pdf_to_images(filepath_or_stream=pdf_stream, resolution=100)
        assert len(images) == 1
        assert type(images[0]) == np.ndarray

    def test_read_pdf_error_format(self, txt_filepath):
        with pytest.raises(ValueError):
            read_pdf_to_images(filepath_or_stream=txt_filepath, resolution=100)
