"""
Noise tests
"""

import numpy as np
import pytest

from otary.image.image import Image


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
