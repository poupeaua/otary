import numpy as np
from typing import Literal
from otary.image import Image
from otary.image.components.transformer.components.binarizer.ops.niblack_like import (
    threshold_niblack_like,
)


import pytest


class TestThresholdNiblackLike:

    def test_threshold_unknown_method(self):
        with pytest.raises(ValueError):
            img = Image.from_fillvalue(shape=(5, 5), value=127)
            threshold_niblack_like(img.asarray, method="not_an_expected_binary_method")


class TestThresholdSimple:

    @pytest.mark.parametrize("thresh", [25, 50, 100, 150])
    def test_threshold_simple(
        self, thresh: Literal[25] | Literal[50] | Literal[100] | Literal[150]
    ):
        img = Image.from_fillvalue(shape=(5, 5), value=thresh - 1)
        img.asarray[0, 0] = 255
        img.threshold_simple(thresh=thresh)
        assert img.asarray[0, 0] == 255
        img.asarray[0, 0] = 0
        assert np.all(img.asarray == 0)


class TestThresholdAdaptive:

    def test_threshold_adaptive_basic(self):
        img = Image.from_fillvalue(shape=(5, 5), value=127)
        img.threshold_adaptive()
        assert np.all((img.asarray == 0) | (img.asarray == 255))

    def test_threshold_adaptive_uniform_image(self):
        img = Image.from_fillvalue(shape=(5, 5), value=200)
        img.threshold_adaptive()
        assert np.all(img.asarray == 255)

    def test_threshold_adaptive_low_values(self):
        img = Image.from_fillvalue(shape=(5, 5), value=55)
        img.asarray[2, 2] = 200
        img.threshold_adaptive()
        assert img.asarray[2, 2] == 255
        assert img.asarray[0, 0] == 0
        assert img.asarray[4, 4] == 0

    def test_threshold_adaptive_mixed_values(self):
        img = Image.from_fillvalue(shape=(5, 5), value=127)
        img.asarray[0, 0] = 200
        img.asarray[4, 4] = 50
        img.threshold_adaptive()
        assert img.asarray[0, 0] == 255
        assert img.asarray[4, 4] == 0


class TestThresholdOtsu:

    def test_threshold_otsu(self):
        img = Image.from_fillvalue(shape=(5, 5), value=200)
        img.threshold_otsu()
        assert np.all(img.asarray == 255)


class TestThresholdSauvola:

    def test_threshold_sauvola(self):
        img = Image.from_fillvalue(shape=(5, 5), value=127)
        img.threshold_sauvola()
        assert np.all(img.asarray == 255)


class TestThresholdBradley:

    def test_threshold_bradley(self):
        img = Image.from_fillvalue(shape=(5, 5), value=127)
        img.asarray[0, 0] = 200
        img.asarray[4, 4] = 50
        img.threshold_bradley()
        assert img.asarray[0, 0] == 255
        assert img.asarray[4, 4] == 0
