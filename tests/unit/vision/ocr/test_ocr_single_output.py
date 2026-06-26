"""
Test the OcrSingleOutput class.
"""

import pytest
import re

from otary.vision.ocr import OcrSingleOutput
import otary.geometry as geo


@pytest.fixture
def sample_bbox():
    return geo.Rectangle.unit()


@pytest.fixture
def ocr_output_with_text(sample_bbox) -> OcrSingleOutput:
    return OcrSingleOutput(bbox=sample_bbox, text="This is a sample OCR output text.")


@pytest.fixture
def ocr_output_without_text(sample_bbox) -> OcrSingleOutput:
    return OcrSingleOutput(bbox=sample_bbox, text=None)


class TestConstainsStringOcrSingleOutput:

    def test_contains_string_any_single_match(
        self, ocr_output_with_text: OcrSingleOutput
    ):
        assert ocr_output_with_text.contains_string("sample", cond="any") is True

    def test_contains_string_any_no_match(self, ocr_output_with_text: OcrSingleOutput):
        assert ocr_output_with_text.contains_string("missing", cond="any") is False

    def test_contains_string_any_multiple_matches(
        self, ocr_output_with_text: OcrSingleOutput
    ):
        assert (
            ocr_output_with_text.contains_string(["sample", "OCR"], cond="any") is True
        )

    def test_contains_string_all_matches(self, ocr_output_with_text: OcrSingleOutput):
        assert (
            ocr_output_with_text.contains_string(["sample", "OCR"], cond="all") is True
        )

    def test_contains_string_all_partial_match(
        self, ocr_output_with_text: OcrSingleOutput
    ):
        assert (
            ocr_output_with_text.contains_string(["sample", "missing"], cond="all")
            is False
        )

    def test_contains_string_with_empty_text(
        self, ocr_output_without_text: OcrSingleOutput
    ):
        assert ocr_output_without_text.contains_string("sample", cond="any") is False

    def test_contains_string_invalid_condition(
        self, ocr_output_with_text: OcrSingleOutput
    ):
        with pytest.raises(
            ValueError,
            match="The parameter cond must be in \\[any, all\\]. Found invalid",
        ):
            ocr_output_with_text.contains_string("sample", cond="invalid")


class TestContainsRegexOcrSingleOutput:

    def test_contains_regex_match(self, ocr_output_with_text: OcrSingleOutput):
        assert ocr_output_with_text.contains_regex(r"sample") is True

    def test_contains_regex_no_match(self, ocr_output_with_text: OcrSingleOutput):
        assert ocr_output_with_text.contains_regex(r"missing") is False

    def test_contains_regex_almost_match(self, ocr_output_with_text: OcrSingleOutput):
        assert ocr_output_with_text.contains_regex(r"sampleOCR") is False

    def test_contains_regex_full_match(self, ocr_output_with_text: OcrSingleOutput):
        assert (
            ocr_output_with_text.contains_regex(r"This is a sample OCR output text\.")
            is True
        )

    def test_contains_regex_with_empty_text(
        self, ocr_output_without_text: OcrSingleOutput
    ):
        assert ocr_output_without_text.contains_regex(r"sample") is False

    def test_contains_regex_invalid_regex(self, ocr_output_with_text: OcrSingleOutput):
        with pytest.raises(re.error):
            ocr_output_with_text.contains_regex(r"invalid[")
