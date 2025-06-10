import pytest

from src.image import Image
from src.geometry import Rectangle


class TestTransformerImageCropClipping:

    def test_crop_clipping(self):
        img = Image.from_fillvalue(shape=(5, 5), value=0)
        x0, y0, x1, y1 = 1, 1, 4, 3
        img.crop(x0=x0, y0=y0, x1=x1, y1=y1, clip=True)
        width_expected = x1 - x0
        height_expected = y1 - y0
        assert img.shape_array == (height_expected, width_expected)


class TestTransformerImageCropFromTopleft:

    def test_crop_from_topleft(self):
        img = Image.from_fillvalue(shape=(5, 5), value=0)
        width = 4
        height = 3
        img.crop_from_topleft(topleft=[1, 1], width=width, height=height)
        assert img.shape_array == (height, width)


class TestTransformerImageCropAxisAlignedBbox:

    def test_crop_axis_aligned_bbox(self):
        img = Image.from_fillvalue(shape=(5, 5), value=0)
        bbox = Rectangle.from_topleft_bottomright(topleft=[1, 1], bottomright=[4, 3])
        img.crop_from_axis_aligned_bbox(bbox)
        assert img.shape_array == (3, 4)


class TestTransformerImageCropAroundSegment:

    def test_crop_around_segment_horizontal(self):
        img = Image.from_fillvalue(shape=(5, 5), value=0)
        img, _, _, _ = img.crop_around_segment_horizontal(
            segment=[[1, 2], [3, 2]], dim_crop_rect=(-1, 3), added_width=0
        )
        assert img.shape_array == (3, 2)

    def test_crop_around_segment_horizontal_with_segment_horizontal(self):
        img = Image.from_fillvalue(shape=(5, 5), value=0)
        img = img.crop_around_segment_horizontal_faster(
            segment=[[1, 1], [3, 1]], dim_crop_rect=(-1, 3), added_width=0
        )
        assert img.shape_array == (3, 2)

    def test_crop_around_segment_horizontal_with_segment_vertical(self):
        img = Image.from_fillvalue(shape=(5, 5), value=0)
        img = img.crop_around_segment_horizontal_faster(
            segment=[[1, 1], [1, 3]], dim_crop_rect=(-1, 3), added_width=0
        )
        assert img.shape_array == (3, 2)


class TestTransformerImageCropNextToRectangle:

    def test_crop_next_to_rectangle_default_params(self):
        img = Image.from_fillvalue(shape=(10, 10), value=0)
        rect = Rectangle.from_topleft_bottomright(topleft=[3, 3], bottomright=[5, 5])
        cropped_img = img.crop_next_to_rectangle(rect=rect, rect_topleft_ix=0)
        assert cropped_img.shape_array == (2, 2)

    def test_crop_next_to_rectangle_custom_crop_dim(self):
        img = Image.from_fillvalue(shape=(10, 10), value=0)
        rect = Rectangle.from_topleft_bottomright(topleft=[3, 3], bottomright=[5, 5])
        cropped_img = img.crop_next_to_rectangle(
            rect=rect, rect_topleft_ix=0, crop_dim=(3, 3)
        )
        assert cropped_img.shape_array == (3, 3)

    def test_crop_next_to_rectangle_with_crop_shift(self):
        img = Image.from_fillvalue(shape=(10, 10), value=0)
        img.asarray[4, 7] = 7
        img.asarray[4, 6] = 6
        img.asarray[4, 5] = 5
        img.asarray[4, 4] = 4
        img.asarray[4, 3] = 3
        rect = Rectangle.from_topleft_bottomright(topleft=[3, 3], bottomright=[5, 5])
        cropped_img = img.crop_next_to_rectangle(
            rect=rect, rect_topleft_ix=0, crop_shift=(3, -1), crop_dim=(3, 3)
        )
        assert cropped_img.asarray[1, 1] == 5
        assert cropped_img.asarray[2, 2] == 0

    def test_crop_next_to_rectangle_invalid_crop_dim(self):
        img = Image.from_fillvalue(shape=(10, 10), value=0)
        rect = Rectangle.from_topleft_bottomright(topleft=[3, 3], bottomright=[5, 5])
        with pytest.raises(AssertionError):
            img.crop_next_to_rectangle(rect=rect, rect_topleft_ix=0, crop_dim=(-1, -5))

    def test_crop_next_to_rectangle_large_crop_dim(self):
        img = Image.from_fillvalue(shape=(10, 10), value=0)
        rect = Rectangle.from_topleft_bottomright(topleft=[3, 3], bottomright=[5, 5])
        cropped_img = img.crop_next_to_rectangle(
            rect=rect, rect_topleft_ix=0, crop_dim=(10, 10)
        )
        assert cropped_img.shape_array == (10, 10)
