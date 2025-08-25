"""
Test suite for intensity utilities.
"""

import numpy as np

from otary.image.utils.intensity import intensity_local, intensity_local_v2


class TestIntensityUtils:

    def test_intensity_v1_equal_v2_int(self):
        np.random.seed(0)
        im = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)

        assert np.array_equal(
            intensity_local(im, window_size=15, cast_int=True),
            intensity_local_v2(im, window_size=15, cast_int=True),
        )

    def test_intensity_v1_equal_v2_float(self):
        np.random.seed(0)
        im = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)

        assert np.allclose(
            intensity_local(im, window_size=15, cast_int=False),
            intensity_local_v2(im, window_size=15, cast_int=False),
        )

    def test_intensity_v1_equal_v2_float_non_normalized(self):
        np.random.seed(0)
        im = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)

        assert np.array_equal(
            intensity_local(im, window_size=15, cast_int=False, normalize=False),
            intensity_local_v2(im, window_size=15, cast_int=False, normalize=False),
        )
