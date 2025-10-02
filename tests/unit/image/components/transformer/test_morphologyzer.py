import pytest
import numpy as np

from otary.image import Image


class TestMorphologyzerImageResizeFixed:

    def test_resize_fixed(self):
        img = Image.from_fillvalue(shape=(5, 5), value=0)
        new_shape = (10, 15)
        img.resize_fixed(dim=new_shape)
        assert img.shape_array == (new_shape[1], new_shape[0])

    def test_resize_fixed_normal(self):
        img = Image.from_fillvalue(shape=(8, 6), value=100)
        new_dim = (16, 12)
        img.resize_fixed(dim=new_dim)
        assert img.shape_array == (new_dim[1], new_dim[0])

    def test_resize_fixed_negative_height(self):
        img = Image.from_fillvalue(shape=(10, 20), value=50)
        new_dim = (40, -1)
        img.resize_fixed(dim=new_dim)
        assert img.shape_array[0] == int(10 * (40 / 20))
        assert img.shape_array[1] == 40

    def test_resize_fixed_negative_width(self):
        img = Image.from_fillvalue(shape=(10, 20), value=50)
        new_dim = (-1, 40)
        img.resize_fixed(dim=new_dim)
        assert img.shape_array[0] == 40
        assert img.shape_array[1] == int(20 * (40 / 10))

    def test_resize_fixed_both_negative_raises(self):
        img = Image.from_fillvalue(shape=(5, 5), value=0)
        with pytest.raises(ValueError):
            img.resize_fixed(dim=(-1, -1))

    def test_resize_fixed_copy_true_returns_new_image(self):
        img = Image.from_fillvalue(shape=(5, 5), value=0)
        new_dim = (10, 10)
        result = img.resize_fixed(dim=new_dim, copy=True)
        assert result is not None
        assert hasattr(result, "asarray")
        assert result.asarray.shape == (10, 10)
        # Original image should remain unchanged
        assert img.asarray.shape == (5, 5)


class TestMorphologyzerImageResizeFactor:

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


class TestMorphologyzerImageAddNoiseSaltAndPepper:

    def test_add_noise_salt_and_pepper_error(self):
        with pytest.raises(ValueError):
            img = Image.from_fillvalue(shape=(5, 5), value=128)
            img.add_noise_salt_and_pepper(amount=1.1)

    def test_add_noise_salt_and_pepper_changes_pixels(self):
        img = Image.from_fillvalue(shape=(20, 20), value=128)
        original = img.asarray.copy()
        img.add_noise_salt_and_pepper(amount=0.1)
        # Should have some pixels set to 0 and some to 255
        assert np.any(img.asarray == 0)
        assert np.any(img.asarray == 255)
        # Should not be identical to original
        assert not np.array_equal(img.asarray, original)

    def test_add_noise_salt_and_pepper_amount_zero(self):
        img = Image.from_fillvalue(shape=(10, 10), value=50)
        original = img.asarray.copy()
        img.add_noise_salt_and_pepper(amount=0.0)
        # No pixels should be changed
        assert np.array_equal(img.asarray, original)

    def test_add_noise_salt_and_pepper_full_amount(self):
        img = Image.from_fillvalue(shape=(5, 5), value=100)
        img.add_noise_salt_and_pepper(amount=1.0)
        # All pixels should be either 0 or 255
        assert np.all(np.isin(img.asarray, [0, 255]))

    def test_add_noise_salt_and_pepper_on_multichannel(self):
        img = Image(np.ones((10, 10, 3), dtype=np.uint8) * 100)
        img.add_noise_salt_and_pepper(amount=0.2)
        # Should have some pixels set to 0 and some to 255 in at least one channel
        assert np.any(img.asarray == 0)
        assert np.any(img.asarray == 255)


class TestMorphologyzerImageAddNoiseGaussian:
    def test_add_noise_gaussian_changes_pixels(self):
        img = Image.from_fillvalue(shape=(20, 20), value=128)
        original = img.asarray.copy()
        img.add_noise_gaussian(mean=0, std=10)
        # Should not be identical to original
        assert not np.array_equal(img.asarray, original)
        # Values should be clipped between 0 and 255
        assert np.all((img.asarray >= 0) & (img.asarray <= 255))

    def test_add_noise_gaussian_zero_std(self):
        img = Image.from_fillvalue(shape=(10, 10), value=50)
        original = img.asarray.copy()
        img.add_noise_gaussian(mean=0, std=0)
        # No pixels should be changed
        assert np.array_equal(img.asarray, original)

    def test_add_noise_gaussian_high_std(self):
        img = Image.from_fillvalue(shape=(5, 5), value=100)
        img.add_noise_gaussian(mean=0, std=100)
        # All pixels should be between 0 and 255
        assert np.all((img.asarray >= 0) & (img.asarray <= 255))

    def test_add_noise_gaussian_on_multichannel(self):
        img = Image(np.ones((10, 10, 3), dtype=np.uint8) * 100)
        original = img.asarray.copy()
        img.add_noise_gaussian(mean=0, std=20)
        # Should not be identical to original
        assert not np.array_equal(img.asarray, original)
        # Should have shape unchanged
        assert img.asarray.shape == original.shape
