from typing import Literal
import pytest
import numpy as np

from otary.geometry.discrete.shape.axis_aligned_rectangle import AxisAlignedRectangle
from otary.image import Image


@pytest.fixture
def im_pdf_crop() -> Image:
    """Fixture to produce an image of text from a sample pdf that can be
    used to verify the quality of binarization processes

    Returns:
        Image: output image
    """
    aabb = AxisAlignedRectangle.from_topleft_bottomright([0.3, 0.125], [0.7, 0.225])
    im = Image.from_pdf("tests/data/test.pdf", resolution=1000, clip_pct=aabb)
    return im


class TestThresholdNiblack:

    def test_threshold_niblack_basic(self):
        img = Image.from_fillvalue(shape=(5, 5), value=127)
        img.transformer.binarizer.threshold_niblack()
        assert np.all((img.asarray == 0) | (img.asarray == 255))

    def test_threshold_niblack_window_size_even(self):
        img = Image.from_fillvalue(shape=(5, 5), value=127)
        img.transformer.binarizer.threshold_niblack(window_size=10)
        assert np.all((img.asarray == 0) | (img.asarray == 255))

    def test_threshold_niblack_low_values(self):
        img = Image.from_fillvalue(shape=(5, 5), value=55)
        img.asarray[2, 2] = 200
        img.transformer.binarizer.threshold_niblack()
        assert img.asarray[2, 2] == 255
        assert img.asarray[0, 0] == 0
        assert img.asarray[4, 4] == 0

    def test_threshold_niblack_mixed_values(self):
        img = Image.from_fillvalue(shape=(5, 5), value=127)
        img.asarray[0, 0] = 200
        img.asarray[4, 4] = 50
        img.transformer.binarizer.threshold_niblack()
        assert img.asarray[0, 0] == 255
        assert img.asarray[4, 4] == 0

    @pytest.mark.parametrize("window_size, k", [(15, 0.2), (25, 0.5), (5, 0.1)])
    def test_threshold_niblack_parametrized(
        self, window_size: Literal[15] | Literal[25] | Literal[5], k: float
    ):
        img = Image.from_fillvalue(shape=(5, 5), value=127)
        img.transformer.binarizer.threshold_niblack(window_size=window_size, k=k)
        assert np.all((img.asarray == 0) | (img.asarray == 255))


class TestThresholdISauvola:

    def test_threshold_isauvola(self):
        img = Image.from_fillvalue(shape=(5, 5), value=127)
        img.asarray[0, 0] = 200
        img.asarray[4, 4] = 50
        img.transformer.binarizer.threshold_isauvola()
        assert img.asarray[0, 0] == 255
        assert img.asarray[4, 4] == 0

    def test_threshold_isauvola_opening(self):
        img = Image.from_fillvalue(shape=(5, 5), value=255)
        img.asarray[0, 0] = 0
        img.asarray[4, 4] = 0
        img.transformer.binarizer.threshold_isauvola(opening_n_min_pixels=1)
        assert np.unique(img.asarray).size == 2


class TestThresholdFeng:

    def test_threshold_feng(self):
        img = Image.from_fillvalue(shape=(5, 5), value=127)
        img.asarray[0, 0] = 200
        img.asarray[4, 4] = 50
        img.transformer.binarizer.threshold_feng()
        assert img.asarray[0, 0] == 255
        assert img.asarray[4, 4] == 0


class TestThresholdSu:

    def test_threshold_su(self):
        img = Image.from_fillvalue(shape=(5, 5), value=127)
        img.asarray[0, 0] = 200
        img.asarray[4, 4] = 50
        img.transformer.binarizer.threshold_su()
        assert img.asarray[0, 0] == 255
        assert img.asarray[4, 4] == 0


class TestThresholdPhansalkar:

    def test_threshold_phansalkar(self):
        img = Image.from_fillvalue(shape=(5, 5), value=127)
        img.asarray[0, 0] = 200
        img.asarray[4, 4] = 50
        img.transformer.binarizer.threshold_phansalkar()
        assert img.asarray[0, 0] == 255
        assert img.asarray[4, 4] == 0


class TestThresholdWolf:

    def test_threshold_wolf(self):
        img = Image.from_fillvalue(shape=(5, 5), value=127)
        img.asarray[0, 0] = 200
        img.asarray[4, 4] = 50
        img.transformer.binarizer.threshold_wolf()
        assert img.asarray[0, 0] == 255
        assert img.asarray[4, 4] == 0


class TestThresholdBernsen:

    def test_threshold_bernsen(self):
        img = Image.from_fillvalue(shape=(5, 5), value=127)
        img.asarray[0, 0] = 200
        img.asarray[4, 4] = 50
        img.transformer.binarizer.threshold_bernsen()
        assert img.asarray[0, 0] == 255
        assert img.asarray[4, 4] == 0


class TestThresholdWan:

    def test_threshold_wan(self):
        img = Image.from_fillvalue(shape=(5, 5), value=127)
        img.asarray[0, 0] = 200
        img.asarray[4, 4] = 50
        img.transformer.binarizer.threshold_wan()
        assert img.asarray[0, 0] == 255
        assert img.asarray[4, 4] == 0


class TestThresholdNick:

    def test_threshold_nick(self):
        img = Image.from_fillvalue(shape=(5, 5), value=127)
        img.asarray[0, 0] = 200
        img.asarray[4, 4] = 50
        img.transformer.binarizer.threshold_nick(window_size=20)
        assert img.asarray[0, 0] == 255
        assert img.asarray[4, 4] == 0


class TestThresholdSingh:

    def test_threshold_singh(self):
        img = Image.from_fillvalue(shape=(5, 5), value=127)
        img.asarray[0, 0] = 200
        img.asarray[4, 4] = 50
        img.transformer.binarizer.threshold_singh()
        assert img.asarray[0, 0] == 255
        assert img.asarray[4, 4] == 0


class TestThresholdGatos:

    def test_threshold_gatos(self, im_pdf_crop: Image):
        img = im_pdf_crop
        img.transformer.binarizer.threshold_gatos()
        assert img.asarray[0, 0] == 255
        assert img.asarray[53, 36] == 0

    def test_threshold_gatos_upsampling(self, im_pdf_crop: Image):
        img = im_pdf_crop
        img.transformer.binarizer.threshold_gatos(upsampling=True)
        assert img.asarray[0, 0] == 255
        assert img.asarray[53, 36] == 0

    def test_threshold_gatos_upsampling_negative_integer(self, im_pdf_crop: Image):
        with pytest.raises(ValueError):
            img = im_pdf_crop
            img.transformer.binarizer.threshold_gatos(
                upsampling=True, upsampling_factor=-2
            )

    def test_threshold_gatos_upsampling_non_integer(self, im_pdf_crop: Image):
        with pytest.raises(ValueError):
            img = im_pdf_crop
            img.transformer.binarizer.threshold_gatos(
                upsampling=True, upsampling_factor="random_string"
            )

    def test_threshold_gatos_upsampling_p1_error(self, im_pdf_crop: Image):
        with pytest.raises(ValueError):
            img = im_pdf_crop
            img.transformer.binarizer.threshold_gatos(p1=1.2)


class TestThresholdAdOtsu:

    def test_threshold_adotsu(self, im_pdf_crop: Image):
        img = im_pdf_crop
        img.transformer.binarizer.threshold_adotsu()
        assert img.asarray[0, 0] == 255
        assert img.asarray[52, 35] == 0


class TestThresholdFair:

    def test_threshold_fair_otsu(self, im_pdf_crop: Image):
        img = im_pdf_crop
        img.transformer.binarizer.threshold_fair(
            sfair_clustering_algo="otsu",
            sfair_window_size=15,
            sfair_clustering_max_iter=15,
        )
        assert img.asarray[0, 0] == 255
        assert img.asarray[52, 35] == 0

    def test_threshold_fair_em(self, im_pdf_crop: Image):

        img = im_pdf_crop
        img.transformer.binarizer.threshold_fair(
            sfair_clustering_algo="em",
            sfair_window_size=60,
            sfair_clustering_max_iter=15,
        )
        assert img.asarray[0, 0] == 255
        assert img.asarray[52, 35] == 0

    def test_threshold_fair_kmeans(self, im_pdf_crop: Image):
        img = im_pdf_crop
        img.transformer.binarizer.threshold_fair(
            sfair_clustering_algo="kmeans",
            sfair_window_size=11,
            sfair_clustering_max_iter=15,
        )
        assert img.asarray[0, 0] == 255
        assert img.asarray[52, 35] == 0

    def test_threshold_fair_unknown_clustering(self):
        with pytest.raises(ValueError):
            img = Image.from_fillvalue(shape=(5, 5), value=127)
            img.transformer.binarizer.threshold_fair(sfair_clustering_algo="unkown")
