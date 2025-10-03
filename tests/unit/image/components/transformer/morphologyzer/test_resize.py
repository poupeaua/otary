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
