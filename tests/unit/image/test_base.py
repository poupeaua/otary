"""
Unit tests BaseImage class
"""

import numpy as np

from otary.image import Image


class TestBaseImageGlobalMethods:

    def test_init_image_from_array(self):
        img = Image(image=np.full(shape=(5, 5, 3), fill_value=0))
        assert len(img.shape_array) == 3


class TestImageAsArray:

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


class TestImageCenter:

    def test_center(self):
        shape = (20, 10)
        img = Image.from_fillvalue(shape=shape, value=0)
        assert np.array_equal(img.center, np.array([shape[1], shape[0]]) / 2)


class TestImageNormSideLength:

    def test_norm_side_length(self):
        shape = (20, 10)
        img = Image.from_fillvalue(shape=shape, value=0)
        assert img.norm_side_length == int(np.sqrt(shape[0] * shape[1]))


class TestImageAsArrayBinary:

    def test_asarray_norm_white(self):
        img = Image.from_fillvalue(shape=(5, 5), value=255)
        assert np.all(img.asarray_binary == 1)

    def test_asarray_norm_grey(self):
        img = Image.from_fillvalue(shape=(5, 5), value=127)
        assert np.all(np.round(img.asarray_binary, 3) == 0.498)

    def test_asarray_norm_black(self):
        img = Image.from_fillvalue(shape=(5, 5), value=0)
        assert np.all(img.asarray_binary == 0)


class TestImageDistPct:

    def test_dist_pct(self):
        pct = 0.4
        shape = (20, 10)
        img = Image.from_fillvalue(shape=shape, value=0)
        assert img.dist_pct(pct=pct) == int(np.sqrt(shape[0] * shape[1])) * pct


class TestImageCopy:

    def test_copy(self):
        img = Image.from_fillvalue(shape=(5, 5), value=0)
        assert np.array_equal(img.asarray, img.copy().asarray)


class TestImageAsApiFileInput:
    def test_as_api_file_input_default(self):
        img = Image.from_fillvalue(shape=(10, 10), value=128)
        result = img.as_api_file_input()
        assert isinstance(result, dict)
        assert "file" in result
        file_tuple = result["file"]
        assert isinstance(file_tuple, tuple)
        assert len(file_tuple) == 3
        filename, file_bytes, content_type = file_tuple
        assert filename.startswith("image")
        assert isinstance(file_bytes, bytes)
        assert content_type == "image/png"

    def test_as_api_file_input_jpeg(self):
        img = Image.from_fillvalue(shape=(10, 10), value=128)
        result = img.as_api_file_input(fmt="jpeg", filename="testfile")
        assert "file" in result
        filename, file_bytes, content_type = result["file"]
        assert filename == "testfile.jpeg"
        assert isinstance(file_bytes, bytes)
        assert content_type == "image/jpeg"


class TestImageChannels:
    def test_channels_grayscale(self):
        img = Image.from_fillvalue(shape=(10, 10), value=128)
        # Convert to grayscale explicitly to ensure single channel
        img.as_grayscale()
        assert img.channels == 1

    def test_channels_colorscale(self):
        img = Image.from_fillvalue(shape=(10, 10, 3), value=128)
        assert img.channels == 3

    def test_channels_after_conversion(self):
        img = Image.from_fillvalue(shape=(10, 10), value=128)
        # Initially grayscale
        assert img.channels == 1
        # Convert to colorscale
        img.as_colorscale()
        assert img.channels == 3
        # Convert back to grayscale
        img.as_grayscale()
        assert img.channels == 1


class TestImageShape:
    def test_shape_colorscale(self):
        img = Image.from_fillvalue(shape=(10, 20, 3), value=128)
        # shape property should return (width, height, channels)
        assert img.shape_xy == (20, 10, 3)

    def test_shape_grayscale(self):
        img = Image.from_fillvalue(shape=(15, 25), value=128)
        # shape property should return (width, height, channels)
        # For grayscale, channels should be 1
        assert img.shape_xy == (25, 15, 1)

    def test_shape_after_colorscale_conversion(self):
        img = Image.from_fillvalue(shape=(8, 12), value=128)
        img.as_colorscale()
        assert img.shape_xy == (12, 8, 3)

    def test_shape_after_grayscale_conversion(self):
        img = Image.from_fillvalue(shape=(8, 12, 3), value=128)
        img.as_grayscale()
        assert img.shape_xy == (12, 8, 1)


class TestImageWidth:
    def test_width_colorscale(self):
        img = Image.from_fillvalue(shape=(10, 20, 3), value=128)
        # shape is (height, width, channels), so width should be 20
        assert img.width == 20

    def test_width_grayscale(self):
        img = Image.from_fillvalue(shape=(15, 25), value=128)
        # shape is (height, width), so width should be 25
        assert img.width == 25

    def test_width_after_colorscale_conversion(self):
        img = Image.from_fillvalue(shape=(8, 12), value=128)
        img.as_colorscale()
        assert img.width == 12

    def test_width_after_grayscale_conversion(self):
        img = Image.from_fillvalue(shape=(8, 12, 3), value=128)
        img.as_grayscale()
        assert img.width == 12


class TestImageHeight:
    def test_height_colorscale(self):
        img = Image.from_fillvalue(shape=(10, 20, 3), value=128)
        # shape is (height, width, channels), so height should be 10
        assert img.height == 10

    def test_height_grayscale(self):
        img = Image.from_fillvalue(shape=(15, 25), value=128)
        # shape is (height, width), so height should be 15
        assert img.height == 15

    def test_height_after_colorscale_conversion(self):
        img = Image.from_fillvalue(shape=(8, 12), value=128)
        img.as_colorscale()
        assert img.height == 8

    def test_height_after_grayscale_conversion(self):
        img = Image.from_fillvalue(shape=(8, 12, 3), value=128)
        img.as_grayscale()
        assert img.height == 8


class TestImageArea:
    def test_area_colorscale(self):
        img = Image.from_fillvalue(shape=(10, 20, 3), value=128)
        # Area should be height * width
        assert img.area == 10 * 20

    def test_area_grayscale(self):
        img = Image.from_fillvalue(shape=(15, 25), value=128)
        assert img.area == 15 * 25

    def test_area_after_colorscale_conversion(self):
        img = Image.from_fillvalue(shape=(8, 12), value=128)
        img.as_colorscale()
        assert img.area == 8 * 12

    def test_area_after_grayscale_conversion(self):
        img = Image.from_fillvalue(shape=(8, 12, 3), value=128)
        img.as_grayscale()
        assert img.area == 8 * 12

    def test_area_single_pixel(self):
        img = Image.from_fillvalue(shape=(1, 1), value=128)
        assert img.area == 1


class TestImageShapeArray:
    def test_shape_array_colorscale(self):
        img = Image.from_fillvalue(shape=(10, 20, 3), value=128)
        # shape_array should return (height, width, channels)
        assert img.shape_array == (10, 20, 3)

    def test_shape_array_grayscale(self):
        img = Image.from_fillvalue(shape=(15, 25), value=128)
        # For grayscale, shape_array should return (height, width, 1)
        assert img.shape_array == (15, 25)

    def test_shape_array_after_colorscale_conversion(self):
        img = Image.from_fillvalue(shape=(8, 12), value=128)
        img.as_colorscale()
        assert img.shape_array == (8, 12, 3)

    def test_shape_array_after_grayscale_conversion(self):
        img = Image.from_fillvalue(shape=(8, 12, 3), value=128)
        img.as_grayscale()
        assert img.shape_array == (8, 12)

    def test_shape_array_single_pixel(self):
        img = Image.from_fillvalue(shape=(1, 1), value=128)
        assert img.shape_array == (1, 1)


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


class TestImageIsGray:
    def test_is_gray_true_for_grayscale(self):
        img = Image.from_fillvalue(shape=(10, 10), value=128)
        assert img.is_gray is True

    def test_is_gray_false_for_colorscale(self):
        img = Image.from_fillvalue(shape=(10, 10, 3), value=128)
        assert img.is_gray is False

    def test_is_gray_after_as_grayscale(self):
        img = Image.from_fillvalue(shape=(10, 10, 3), value=128)
        img.as_grayscale()
        assert img.is_gray is True

    def test_is_gray_after_as_colorscale(self):
        img = Image.from_fillvalue(shape=(10, 10), value=128)
        img.as_colorscale()
        assert img.is_gray is False

    def test_is_gray_single_pixel_grayscale(self):
        img = Image.from_fillvalue(shape=(1, 1), value=128)
        assert img.is_gray is True

    def test_is_gray_single_pixel_colorscale(self):
        img = Image.from_fillvalue(shape=(1, 1, 3), value=128)
        assert img.is_gray is False


class TestImageAsBlack:

    def test_as_black_grayscale(self):
        img = Image.from_fillvalue(shape=(5, 5), value=128)
        img.as_black()
        assert np.all(img.asarray == 0)
        assert img.asarray.shape == (5, 5)

    def test_as_black_colorscale(self):
        img = Image.from_fillvalue(shape=(4, 3, 3), value=128)
        img.as_black()
        assert np.all(img.asarray == 0)
        assert img.asarray.shape == (4, 3, 3)

    def test_as_black_preserves_shape(self):
        shape = (7, 8)
        img = Image.from_fillvalue(shape=shape, value=77)
        img.as_black()
        assert img.asarray.shape == shape

    def test_as_black_after_as_white(self):
        img = Image.from_fillvalue(shape=(2, 2), value=0)
        img.as_white()
        img.as_black()
        assert np.all(img.asarray == 0)


class TestBaseImageAsGrayscale:

    def test_as_grayscale(self):
        img = Image.from_fillvalue(shape=(5, 5, 3), value=0).as_grayscale()
        assert len(img.shape_array) == 2

    def test_as_grayscale_nochange(self):
        img = Image.from_fillvalue(shape=(5, 5), value=0).as_grayscale()
        assert len(img.shape_array) == 2


class TestBaseImageAsColorscale:

    def test_as_colorscale(self):
        img = Image.from_fillvalue(shape=(5, 5), value=0).as_colorscale()
        assert len(img.shape_array) == 3

    def test_as_colorscale_nochange(self):
        img = Image.from_fillvalue(shape=(5, 5, 3), value=0).as_colorscale()
        assert len(img.shape_array) == 3


class TestImageAsWhite:

    def test_as_white(self):
        img = Image.from_fillvalue(shape=(5, 5), value=0).as_white()
        assert np.all(img.asarray == 255)


class TestImageAsPIL:

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


class TestImageAsFilled:
    def test_as_filled_default_white(self):
        img = Image.from_fillvalue(shape=(5, 5), value=0)
        img.as_filled()
        assert (img.asarray == 255).all()

    def test_as_filled_custom_value(self):
        img = Image.from_fillvalue(shape=(3, 4), value=0)
        img.as_filled(fill_value=128)
        assert (img.asarray == 128).all()

    def test_as_filled_colorscale(self):
        img = Image.from_fillvalue(shape=(2, 2, 3), value=0)
        img.as_filled(fill_value=64)
        assert (img.asarray == 64).all()
        assert img.asarray.shape == (2, 2, 3)

    def test_as_filled_with_ndarray(self):
        fill = np.array([10, 20, 30], dtype=np.uint8)
        img = Image.from_fillvalue(shape=(2, 2, 3), value=0)
        img.as_filled(fill_value=fill)
        assert (img.asarray == fill).all()

    def test_as_filled_preserves_shape(self):
        img = Image.from_fillvalue(shape=(7, 8), value=0)
        shape_before = img.asarray.shape
        img.as_filled(fill_value=77)
        assert img.asarray.shape == shape_before


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


class TestImageCorners:
    def test_corners_base(self):
        img = Image.from_fillvalue(shape=(10, 20, 3), value=128)
        corners = img.corners
        # Should be 4 corners, each with 2 coordinates (x, y)
        assert corners.shape == (4, 2)
        # Top-left
        assert (corners[0] == [0, 0]).all()
        # Top-right
        assert (corners[1] == [19, 0]).all()
        # Bottom-right
        assert (corners[2] == [19, 9]).all()
        # Bottom-left
        assert (corners[3] == [0, 9]).all()
