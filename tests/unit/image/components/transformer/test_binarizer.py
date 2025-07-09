import pytest
import numpy as np

from otary.image import Image


class TestTransformerThresholdGeneral:

    @pytest.mark.parametrize("thresh", [25, 50, 100, 150])
    def test_threshold_simple(self, thresh):
        img = Image.from_fillvalue(shape=(5, 5), value=thresh - 1)
        img.asarray[0, 0] = 255
        img.threshold_simple(thresh=thresh)
        assert img.asarray[0, 0] == 255
        img.asarray[0, 0] = 0
        assert np.all(img.asarray == 0)

    def test_binary_error_method(self):
        img = Image.from_fillvalue(shape=(5, 5), value=55)
        with pytest.raises(ValueError):
            img.binary(method="not_an_expected_binary_method")


class TestTransformerThresholdAdaptative:

    def test_threshold_adaptative_basic(self):
        img = Image.from_fillvalue(shape=(5, 5), value=127)
        img.threshold_adaptative()
        assert np.all((img.asarray == 0) | (img.asarray == 255))

    def test_threshold_adaptative_uniform_image(self):
        img = Image.from_fillvalue(shape=(5, 5), value=200)
        img.threshold_adaptative()
        assert np.all(img.asarray == 255)

    def test_threshold_adaptative_low_values(self):
        img = Image.from_fillvalue(shape=(5, 5), value=55)
        img.asarray[2, 2] = 200
        img.threshold_adaptative()
        assert img.asarray[2, 2] == 255
        assert img.asarray[0, 0] == 0
        assert img.asarray[4, 4] == 0

    def test_threshold_adaptative_mixed_values(self):
        img = Image.from_fillvalue(shape=(5, 5), value=127)
        img.asarray[0, 0] = 200
        img.asarray[4, 4] = 50
        img.threshold_adaptative()
        assert img.asarray[0, 0] == 255
        assert img.asarray[4, 4] == 0


class TestTransformerThresholdNiblack:

    def test_threshold_niblack_basic(self):
        img = Image.from_fillvalue(shape=(5, 5), value=127)
        img.threshold_niblack()
        assert np.all((img.asarray == 0) | (img.asarray == 255))

    def test_threshold_niblack_low_values(self):
        img = Image.from_fillvalue(shape=(5, 5), value=55)
        img.asarray[2, 2] = 200
        img.threshold_niblack()
        assert img.asarray[2, 2] == 255
        assert img.asarray[0, 0] == 0
        assert img.asarray[4, 4] == 0

    def test_threshold_niblack_mixed_values(self):
        img = Image.from_fillvalue(shape=(5, 5), value=127)
        img.asarray[0, 0] = 200
        img.asarray[4, 4] = 50
        img.threshold_niblack()
        assert img.asarray[0, 0] == 255
        assert img.asarray[4, 4] == 0

    @pytest.mark.parametrize("window_size, k", [(15, 0.2), (25, 0.5), (5, 0.1)])
    def test_threshold_niblack_parametrized(self, window_size, k):
        img = Image.from_fillvalue(shape=(5, 5), value=127)
        img.threshold_niblack(window_size=window_size, k=k)
        assert np.all((img.asarray == 0) | (img.asarray == 255))


class TestTransformerThresholdOtsu:

    def test_threshold_otsu(self):
        img = Image.from_fillvalue(shape=(5, 5), value=200)
        img.threshold_otsu()
        assert np.all(img.asarray == 255)


class TestTransformerSauvolaMethods:

    def test_threshold_sauvola(self):
        img = Image.from_fillvalue(shape=(5, 5), value=127)
        img.threshold_sauvola()
        assert np.all(img.asarray == 255)


class TestTransformerThresholdBinary:

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
