from otary.image import Image


import numpy as np
import pytest


class TestThresholdThresholdBinary:

    def test_binary_error_method(self):
        img = Image.from_fillvalue(shape=(5, 5), value=55)
        with pytest.raises(ValueError):
            img.binary(method="not_an_expected_binary_method")

    def test_binary_sauvola(self):
        img = Image.from_fillvalue(shape=(5, 5), value=55)
        assert np.all(img.binary(method="sauvola") == 1)

    def test_binaryrev_sauvola(self):
        img = Image.from_fillvalue(shape=(5, 5), value=55)
        assert np.all(img.binaryrev(method="sauvola") == 0)

    def test_binary_otsu(self):
        img = Image.from_fillvalue(shape=(5, 5), value=0)
        assert np.all(img.binary(method="otsu") == 0)

    def test_binaryrev_otsu(self):
        img = Image.from_fillvalue(shape=(5, 5), value=0)
        assert np.all(img.binaryrev(method="otsu") == 1)