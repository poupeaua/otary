import pytest
import pymupdf

from otary.image import Image


@pytest.fixture
def pdf_filepath():
    return "./tests/data/test.pdf"


@pytest.fixture
def jpg_filepath():
    return "./tests/data/test.jpg"


@pytest.fixture
def png_filepath():
    return "./tests/data/test.png"


class TestBaseImageFromFillValue:

    def test_init_image_class_method_from_fillvalue(self):
        img = Image.from_fillvalue(shape=(5, 5, 3), value=0)
        assert len(img.shape_array) == 3

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


class TestBaseImageFromFile:

    def test_init_image_class_method_from_jpg(self, jpg_filepath: str):
        img = Image.from_file(filepath=jpg_filepath)
        assert len(img.shape_array) == 3

    def test_init_image_class_method_from_jpg_grayscale(self, jpg_filepath: str):
        img = Image.from_file(filepath=jpg_filepath, as_grayscale=True)
        assert len(img.shape_array) == 2

    def test_init_image_class_method_from_png(self, png_filepath: str):
        img = Image.from_file(filepath=png_filepath)
        assert len(img.shape_array) == 3

    def test_init_image_class_method_from_png_grayscale(self, png_filepath: str):
        img = Image.from_file(filepath=png_filepath, as_grayscale=True)
        assert len(img.shape_array) == 2

    def test_init_image_class_method_from_png_with_resolution(self, png_filepath: str):
        img = Image.from_file(filepath=png_filepath, resolution=32)
        assert img.shape_array[0] == 32

    def test_init_image_class_method_from_png_invalid_file(self):
        with pytest.raises(ValueError):
            Image.from_file(filepath="./tests/data/test.txt")

    def test_init_image_class_method_from_file_pdf(self, pdf_filepath: str):
        img = Image.from_file(filepath=pdf_filepath, resolution=50)
        assert len(img.shape_array) == 3


class TestBaseImageFromPdf:

    def test_init_image_class_method_from_pdf(self, pdf_filepath: str):
        img = Image.from_pdf(filepath=pdf_filepath, resolution=50)
        assert len(img.shape_array) == 3

    def test_init_image_class_method_from_pdf_grayscale(self, pdf_filepath: str):
        img = Image.from_pdf(filepath=pdf_filepath, as_grayscale=True, resolution=50)
        assert len(img.shape_array) == 2

    def test_init_image_class_method_from_pdf_error_page_nb(self, pdf_filepath: str):
        with pytest.raises(IndexError):
            Image.from_pdf(filepath=pdf_filepath, page_nb=1, resolution=50)

    def test_init_image_class_method_from_pdf_neg_page_nb(self, pdf_filepath: str):
        img = Image.from_pdf(filepath=pdf_filepath, page_nb=-1, resolution=50)
        assert len(img.shape_array) == 3

    def test_init_image_class_method_from_pdf_clip(self, pdf_filepath: str):
        img = Image.from_pdf(
            filepath=pdf_filepath,
            resolution=50,
            clip_pct=pymupdf.Rect(x0=0, y0=0, x1=10, y1=10),
        )
        assert len(img.shape_array) == 3
