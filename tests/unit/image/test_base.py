"""
Unit tests BaseImage class
"""

import pytest
import numpy as np

from src.image import Image

@pytest.fixture
def pdf_filepath():
    return "./tests/data/test.pdf"

@pytest.fixture
def jpg_filepath():
    return "./tests/data/test.jpg"


class TestBaseImageFromFillValue:

    def test_init_image_class_method_from_fillvalue(self):
        img = Image.from_fillvalue(shape=(5, 5, 3), value=0)
        assert len(img.shape) == 3

    def test_init_image_class_method_from_fillvalue_error_val_toobig(self):
        with pytest.raises(ValueError):
            Image.from_fillvalue(shape=(5, 5, 3), value=256)

    def test_init_image_class_method_from_fillvalue_error_val_neg(self):
        with pytest.raises(ValueError):
            Image.from_fillvalue(shape=(5, 5, 3), value=-1)

    def test_init_image_class_method_from_fillvalue_error_shape_toolow(self):
        with pytest.raises(ValueError):
            Image.from_fillvalue(shape=(5,), value=0)

    def test_init_image_class_method_from_fillvalue_error_shape_toobig(self):
        with pytest.raises(ValueError):
            Image.from_fillvalue(shape=(5, 2, 5, 5), value=0)

    def test_init_image_class_method_from_fillvalue_error_shape_incorrect(self):
        with pytest.raises(ValueError):
            Image.from_fillvalue(shape=(5, 5, 4), value=0)


class TestBaseImageFromFileImage:

    def test_init_image_class_method_from_jpg(self, jpg_filepath):
        img = Image.from_fileimage(filepath=jpg_filepath)
        assert len(img.shape) == 3

    def test_init_image_class_method_from_jpg_grayscale(self, jpg_filepath):
        img = Image.from_fileimage(filepath=jpg_filepath, as_grayscale=True)
        assert len(img.shape) == 2

    def test_init_image_class_method_from_jpg_error_file_format(self, pdf_filepath):
        with pytest.raises(ValueError):
            Image.from_fileimage(filepath=pdf_filepath)


class TestBaseImageFromPdf:

    def test_init_image_class_method_from_pdf(self, pdf_filepath):
        img = Image.from_pdf(filepath=pdf_filepath, resolution=50)
        assert len(img.shape) == 3

    def test_init_image_class_method_from_pdf_grayscale(self, pdf_filepath):
        img = Image.from_pdf(filepath=pdf_filepath, as_grayscale=True, resolution=50)
        assert len(img.shape) == 2

    def test_init_image_class_method_from_pdf_error_page_nb(self, pdf_filepath):
        with pytest.raises(IndexError):
            Image.from_pdf(filepath=pdf_filepath, page_nb=1, resolution=50)

    def test_init_image_class_method_from_pdf_neg_page_nb(self, pdf_filepath):
        img = Image.from_pdf(filepath=pdf_filepath, page_nb=-1, resolution=50)
        assert len(img.shape) == 3


class TestBaseImageGlobalMethods:

    def test_init_image_from_array(self):
        img = Image(image=np.full(shape=(5, 5, 3), fill_value=0))
        assert len(img.shape) == 3

    def test_init_image_from_other_image(self):
        img_other = Image(image=np.full(shape=(5, 5, 3), fill_value=0))
        img = Image(img_other)
        assert len(img.shape) == 3

    def test_asarray(self):
        shape = (5, 5, 3)
        fill_value = 45
        img = Image.from_fillvalue(shape=shape, value=fill_value)
        assert np.array_equal(img.asarray, np.full(shape=shape, fill_value=fill_value))

    def test_asarray_set(self):
        img = Image.from_fillvalue(shape=(5, 5, 3), value=0)
        arr = np.full(shape=(25, 25, 3), fill_value=255)
        img.asarray = arr
        assert np.array_equal(img.asarray, arr)

    def test_asarray_height(self):
        heigth = 50
        img = Image.from_fillvalue(shape=(heigth, 25), value=0)
        assert img.height == heigth

    def test_asarray_width(self):
        width = 50
        img = Image.from_fillvalue(shape=(25, width), value=0)
        assert img.width == width

    def test_asarray_area(self):
        heigth, width = 10, 7
        img = Image.from_fillvalue(shape=(heigth, width), value=0)
        assert img.area == heigth * width

    def test_asarray_center(self):
        shape = (200, 100)
        img = Image.from_fillvalue(shape=shape, value=0)
        assert img.center == np.array([50, 100])

    def test_asarray_center(self):
        shape = (200, 100)
        img = Image.from_fillvalue(shape=shape, value=0)
        assert np.array_equal(img.center, np.array([50, 100]))

    def test_asarray_norm_side_length(self):
        shape = (200, 100)
        img = Image.from_fillvalue(shape=shape, value=0)
        assert img.norm_side_length == int(np.sqrt(shape[0] * shape[1]))

    def test_asarray_norm_side_length(self):
        shape = (200, 100)
        img = Image.from_fillvalue(shape=shape, value=0)
        assert img.norm_side_length == int(np.sqrt(shape[0] * shape[1]))


class TestBaseImageIsMethods:

    def test_asarray_is_gray_true(self):
        img = Image.from_fillvalue(shape=(5, 5), value=0)
        assert img.is_gray is True

    def test_asarray_is_gray_false(self):
        img = Image.from_fillvalue(shape=(5, 5, 3), value=0)
        assert img.is_gray is False

    def test_asarray_is_shape_equal_true(self):
        shape = (5, 5, 3)
        img0 = Image.from_fillvalue(shape=shape, value=0)
        img1 = Image.from_fillvalue(shape=shape, value=127)
        assert img0.is_equal_shape(other=img1)

    def test_asarray_is_shape_equal_false(self):
        img0 = Image.from_fillvalue(shape=(5, 5, 3), value=0)
        img1 = Image.from_fillvalue(shape=(10, 7, 3), value=127)
        assert not img0.is_equal_shape(other=img1)

    def test_asarray_is_shape_equal_diff_channels_ok(self):
        img0 = Image.from_fillvalue(shape=(5, 5), value=0) # grayscale
        img1 = Image.from_fillvalue(shape=(5, 5, 3), value=127) # colorscale
        assert img0.is_equal_shape(other=img1, consider_channel=False)

    def test_asarray_is_shape_equal_diff_channels_notok(self):
        img0 = Image.from_fillvalue(shape=(5, 5), value=0) # grayscale
        img1 = Image.from_fillvalue(shape=(5, 5, 3), value=127) # colorscale
        assert not img0.is_equal_shape(other=img1)


class TestBaseImageAsMethods:

    def test_asarray_as_grayscale(self):
        img = Image.from_fillvalue(shape=(5, 5, 3), value=0).as_grayscale()
        assert len(img.shape) == 2

    def test_asarray_as_colorscale(self):
        img = Image.from_fillvalue(shape=(5, 5), value=0).as_colorscale()
        assert len(img.shape) == 3

    def test_asarray_as_filled(self):
        val = 127
        img = Image.from_fillvalue(shape=(5, 5), value=0).as_filled(fill_value=val)
        assert np.all(img.asarray == val)

    def test_asarray_as_white(self):
        img = Image.from_fillvalue(shape=(5, 5), value=0).as_white()
        assert np.all(img.asarray == 255)