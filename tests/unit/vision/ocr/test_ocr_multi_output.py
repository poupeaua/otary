"""
Test the OcrMultiOutput class.
"""

import json

import pytest
import numpy as np

from otary.vision.ocr import OcrMultiOutput, OcrSingleOutput
from otary import Rectangle


class TestOCRMOFromPytesseract:

    @pytest.fixture
    def pytesseract_output(self) -> list:
        """Document"""
        return json.load(
            open("tests/data/vision/example1/output_pytesseract.json", "r")
        )

    def test_from_pytesseract(self, pytesseract_output: dict):
        ocrmo = OcrMultiOutput.from_pytesseract(pytesseract_output)
        assert isinstance(ocrmo, OcrMultiOutput)


class TestOCRMOFromEasyOcr:

    @pytest.fixture
    def easyocr_output(self) -> list:
        return [
            ([[0, 0], [1, 0], [1, 1], [0, 1]], "Hello", 0.9),
            ([[2, 2], [3, 2], [3, 3], [2, 3]], "World", 0.8),
            (
                [[4, 4, 4043], [5, 4, "fef"], [5, 5, -111], [4, 5, 45]],
                "Test",
                0.7,
            ),  # invalid
        ]

    @pytest.fixture
    def easyocr_output_decimals(self) -> list:
        return [
            ([[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5]], "Test", 0.95),
        ]

    def test_from_easyocr_with_valid_input(self, easyocr_output: list):
        result = OcrMultiOutput.from_easyocr(easyocr_output)

        # Assertions
        assert len(result.ocrsos) == 2
        assert result.ocrsos[0].text == "Hello"
        assert result.ocrsos[0].confidence == 0.9
        assert np.array_equal(
            result.ocrsos[0].bbox.asarray, [[0, 0], [1, 0], [1, 1], [0, 1]]
        )
        assert result.ocrsos[1].text == "World"
        assert result.ocrsos[1].confidence == 0.8
        assert np.array_equal(
            result.ocrsos[1].bbox.asarray, [[2, 2], [3, 2], [3, 3], [2, 3]]
        )

    def test_from_easyocr_with_empty_input(self):
        easyocr_output = []

        # Call the method
        result = OcrMultiOutput.from_easyocr(easyocr_output)

        # Assertions
        assert len(result.ocrsos) == 0

    def test_from_easyocr_with_cast_int_disabled(self, easyocr_output_decimals: list):
        result = OcrMultiOutput.from_easyocr(
            easyocr_output_decimals, is_bbox_cast_int_enabled=False
        )

        # Assertions
        assert len(result.ocrsos) == 1
        assert result.ocrsos[0].text == "Test"
        assert result.ocrsos[0].confidence == 0.95
        assert np.array_equal(
            result.ocrsos[0].bbox.asarray,
            [[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5]],
        )


class TestOCRMOFromDoctr:

    @pytest.fixture
    def doctr_output_straight_page(self) -> dict:
        return {
            "pages": [
                {
                    "dimensions": [1000, 2000],  # (height, width)
                    "blocks": [
                        {
                            "lines": [
                                {
                                    "words": [
                                        {
                                            "geometry": [[0.1, 0.1], [0.2, 0.2]],
                                            "value": "Hello",
                                            "confidence": 0.95,
                                            "objectness_score": 0.9,
                                        },
                                        {
                                            "geometry": [[0.3, 0.3], [0.4, 0.4]],
                                            "value": "World",
                                            "confidence": 0.85,
                                            "objectness_score": 0.8,
                                        },
                                        {  # invalid one
                                            "geometry": [
                                                [0.3, 0.3, 95],
                                                [0.4, 0.4, -23],
                                            ],
                                            "value": "World",
                                            "confidence": 0.85,
                                            "objectness_score": 0.8,
                                        },
                                    ]
                                }
                            ]
                        }
                    ],
                }
            ]
        }

    @pytest.fixture
    def doctr_output(self) -> dict:
        return {
            "pages": [
                {
                    "dimensions": [1000, 2000],  # (height, width)
                    "blocks": [
                        {
                            "lines": [
                                {
                                    "words": [
                                        {
                                            "geometry": [
                                                [0.1, 0.1],
                                                [0.1, 0.2],
                                                [0.2, 0.2],
                                                [0.2, 0.1],
                                            ],
                                            "value": "Hello",
                                            "confidence": 0.95,
                                            "objectness_score": 0.9,
                                        },
                                        {
                                            "geometry": [
                                                [0.3, 0.3],
                                                [0.3, 0.4],
                                                [0.4, 0.4],
                                                [0.4, 0.3],
                                            ],
                                            "value": "World",
                                            "confidence": 0.85,
                                            "objectness_score": 0.8,
                                        },
                                        {  # invalid one
                                            "geometry": [
                                                [0.3, 0.3, 4],
                                                [0.3, 0.4, 4],
                                                [0.4, 0.4, 4],
                                                [0.4, 0.3, 155],
                                            ],
                                            "value": "World",
                                            "confidence": 0.85,
                                            "objectness_score": 0.8,
                                        },
                                    ]
                                }
                            ]
                        }
                    ],
                }
            ]
        }

    def test_from_doctr_with_valid_input_assume_straight_page_true(
        self, doctr_output_straight_page: dict
    ):
        result = OcrMultiOutput.from_doctr(
            doctr_output_straight_page, assume_straight_pages=True
        )

        # Assertions
        assert len(result.ocrsos) == 2
        assert result.ocrsos[0].text == "Hello"
        assert result.ocrsos[0].confidence == 0.95
        assert result.ocrsos[0].bbox is not None
        assert np.array_equal(
            result.ocrsos[0].bbox.asarray,
            [[200, 100], [400, 100], [400, 200], [200, 200]],
        )
        assert result.ocrsos[1].text == "World"
        assert result.ocrsos[1].confidence == 0.85
        assert result.ocrsos[1].bbox is not None
        assert np.array_equal(
            result.ocrsos[1].bbox.asarray,
            [[600, 300], [800, 300], [800, 400], [600, 400]],
        )

    def test_from_doctr_with_empty_input(self):
        doctr_output = {"pages": [{"dimensions": [1000, 2000], "blocks": []}]}

        # Call the method
        result = OcrMultiOutput.from_doctr(doctr_output, assume_straight_pages=True)

        # Assertions
        assert len(result.ocrsos) == 0

    def test_from_doctr_normal(self, doctr_output: dict):
        result = OcrMultiOutput.from_doctr(doctr_output, assume_straight_pages=False)

        # Assertions
        assert len(result.ocrsos) == 2
        assert result.ocrsos[0].text == "Hello"
        assert result.ocrsos[0].confidence == 0.95
        assert result.ocrsos[0].bbox is not None
        assert np.array_equal(
            result.ocrsos[0].bbox.asarray,
            [[200, 100], [200, 200], [400, 200], [400, 100]],
        )

        assert result.ocrsos[1].text == "World"
        assert result.ocrsos[1].confidence == 0.85
        assert result.ocrsos[1].bbox is not None
        assert np.array_equal(
            result.ocrsos[1].bbox.asarray,
            [[600, 300], [600, 400], [800, 400], [800, 300]],
        )


class TestOCRMOConfidenceMean:

    @pytest.fixture
    def ocrmultioutput_with_confidences(self) -> OcrMultiOutput:

        return OcrMultiOutput(
            ocrsos=[
                OcrSingleOutput(
                    bbox=Rectangle([[0, 0], [1, 0], [1, 1], [0, 1]]),
                    text="Hello",
                    confidence=0.9,
                ),
                OcrSingleOutput(
                    bbox=Rectangle([[2, 2], [3, 2], [3, 3], [2, 3]]),
                    text="World",
                    confidence=0.8,
                ),
                OcrSingleOutput(
                    bbox=Rectangle([[4, 4], [5, 4], [5, 5], [4, 5]]),
                    text=None,
                    confidence=None,
                ),
            ]
        )

    @pytest.fixture
    def ocrmultioutput_empty(self) -> OcrMultiOutput:
        return OcrMultiOutput(ocrsos=[])

    def test_confidence_mean_with_non_empty_data(
        self, ocrmultioutput_with_confidences: OcrMultiOutput
    ):
        result = ocrmultioutput_with_confidences.confidence_mean()
        assert result == pytest.approx((0.9 + 0.8 + 0.0) / 3, rel=1e-6)

    def test_confidence_mean_with_non_empty_data_exclude_none(
        self, ocrmultioutput_with_confidences: OcrMultiOutput
    ):
        result = ocrmultioutput_with_confidences.confidence_mean(count_none=False)
        assert result == pytest.approx((0.9 + 0.8) / 2, rel=1e-6)

    def test_confidence_mean_with_empty_data(
        self, ocrmultioutput_empty: OcrMultiOutput
    ):
        result = ocrmultioutput_empty.confidence_mean()
        assert result == 0


class TestOCRMOWordsIn:

    @pytest.fixture
    def ocrmultioutput_with_words(self) -> OcrMultiOutput:
        return OcrMultiOutput(
            ocrsos=[
                OcrSingleOutput(
                    bbox=Rectangle([[0, 0], [1, 0], [1, 1], [0, 1]]),
                    text="Hello",
                    confidence=0.9,
                ),
                OcrSingleOutput(
                    bbox=Rectangle([[2, 2], [3, 2], [3, 3], [2, 3]]),
                    text="World",
                    confidence=0.8,
                ),
                OcrSingleOutput(
                    bbox=Rectangle([[4, 4], [5, 4], [5, 5], [4, 5]]),
                    text="Test",
                    confidence=0.7,
                ),
            ]
        )

    def test_words_in_with_box_containing_words(
        self, ocrmultioutput_with_words: OcrMultiOutput
    ):
        box = Rectangle([[0, 0], [3, 0], [3, 3], [0, 3]])
        result = ocrmultioutput_with_words.words_in(box)

        # Assertions
        assert len(result) == 2
        assert result[0].text == "Hello"
        assert result[1].text == "World"

    def test_words_in_with_box_containing_no_words(
        self, ocrmultioutput_with_words: OcrMultiOutput
    ):
        box = Rectangle([[6, 6], [7, 6], [7, 7], [6, 7]])
        result = ocrmultioutput_with_words.words_in(box)

        # Assertions
        assert len(result) == 0

    def test_words_in_with_box_extension(
        self, ocrmultioutput_with_words: OcrMultiOutput
    ):
        box = Rectangle([[3, 3], [4, 3], [4, 4], [3, 4]])
        result = ocrmultioutput_with_words.words_in(box, box_expand_scale=3)

        # Assertions
        assert len(result) == 2
        assert result[0].text == "World"
        assert result[1].text == "Test"

    def test_words_in_with_empty_ocrsos(self):
        ocrmultioutput = OcrMultiOutput(ocrsos=[])
        box = Rectangle([[0, 0], [1, 0], [1, 1], [0, 1]])
        result = ocrmultioutput.words_in(box)

        # Assertions
        assert len(result) == 0


class TestOCRMODropDuplicates:

    @pytest.fixture
    def ocrmo_with_dup(self) -> OcrMultiOutput:
        # Two boxes close together, one far away
        bbox1 = Rectangle([[0, 0], [1, 0], [1, 1], [0, 1]], is_cast_int=True)
        bbox2 = Rectangle(
            [[-0.1, -0.1], [2, -0.1], [2, 1.1], [-0.1, 1.1]], is_cast_int=True
        )  # close to bbox1
        bbox3 = Rectangle(
            [[5, 5], [6, 5], [6, 6], [5, 6]], is_cast_int=True
        )  # far from both
        ocrsos = [
            OcrSingleOutput(bbox=bbox1, text="A", confidence=0.9),
            OcrSingleOutput(bbox=bbox2, text="B", confidence=0.8),
            OcrSingleOutput(bbox=bbox3, text="C", confidence=0.7),
        ]
        return OcrMultiOutput(ocrsos=ocrsos)

    def test_drop_duplicates_max_area(self, ocrmo_with_dup: OcrMultiOutput):
        # bbox1 and bbox2 are close, bbox3 is far
        ocrmo = ocrmo_with_dup.copy()
        ocrmo.drop_duplicates(dist_thresh=0.5)
        # Should keep the one with max area among bbox1 and bbox2, and always keep bbox3
        assert len(ocrmo.ocrsos) == 2
        # Both bbox1 and bbox2 have same area, so first one should be kept
        texts = sorted([ocrso.text for ocrso in ocrmo.ocrsos])
        assert texts == ["B", "C"]

    def test_drop_duplicates_min_area(self, ocrmo_with_dup: OcrMultiOutput):
        ocrmo = ocrmo_with_dup.copy()
        ocrmo.drop_duplicates(dist_thresh=0.5)
        # Both bbox1 and bbox2 have same area, so first one should be kept
        assert len(ocrmo.ocrsos) == 2
        texts = sorted([ocrso.text for ocrso in ocrmo.ocrsos])
        assert texts == ["B", "C"]

    def test_drop_duplicates_empty(self):
        ocrmo = OcrMultiOutput(ocrsos=[])
        ocrmo.drop_duplicates(dist_thresh=1.0, criteria="max_area")
        assert len(ocrmo.ocrsos) == 0

    def test_drop_duplicates_invalid_criteria(self, ocrmo_with_dup: OcrMultiOutput):
        ocrmo = ocrmo_with_dup.copy()
        with pytest.raises(ValueError):
            ocrmo.drop_duplicates(dist_thresh=1.0, criteria="invalid_criteria")


class TestOCRMOMerge:

    @pytest.fixture
    def ocrmultioutput_1(self) -> OcrMultiOutput:
        return OcrMultiOutput(
            ocrsos=[
                OcrSingleOutput(
                    bbox=Rectangle([[0, 0], [1, 0], [1, 1], [0, 1]]),
                    text="Hello",
                    confidence=0.9,
                ),
                OcrSingleOutput(
                    bbox=Rectangle([[2, 2], [3, 2], [3, 3], [2, 3]]),
                    text="World",
                    confidence=0.8,
                ),
            ]
        )

    @pytest.fixture
    def ocrmultioutput_2(self) -> OcrMultiOutput:
        return OcrMultiOutput(
            ocrsos=[
                OcrSingleOutput(
                    bbox=Rectangle([[4, 4], [5, 4], [5, 5], [4, 5]]),
                    text="Test",
                    confidence=0.7,
                ),
                OcrSingleOutput(
                    bbox=Rectangle([[6, 6], [7, 6], [7, 7], [6, 7]]),
                    text="Merge",
                    confidence=0.85,
                ),
            ]
        )

    @pytest.fixture
    def ocrmultioutput_empty(self) -> OcrMultiOutput:
        return OcrMultiOutput(ocrsos=[])

    def test_merge_two_non_empty_outputs(
        self, ocrmultioutput_1: OcrMultiOutput, ocrmultioutput_2: OcrMultiOutput
    ):
        result = OcrMultiOutput.merge([ocrmultioutput_1, ocrmultioutput_2])

        assert len(result.ocrsos) == 4
        assert result.ocrsos[0].text == "Hello"
        assert result.ocrsos[1].text == "World"
        assert result.ocrsos[2].text == "Test"
        assert result.ocrsos[3].text == "Merge"

    def test_merge_single_output(self, ocrmultioutput_1: OcrMultiOutput):
        result = OcrMultiOutput.merge([ocrmultioutput_1])

        assert len(result.ocrsos) == 2
        assert result.ocrsos[0].text == "Hello"
        assert result.ocrsos[1].text == "World"

    def test_merge_with_empty_outputs(
        self, ocrmultioutput_1: OcrMultiOutput, ocrmultioutput_empty: OcrMultiOutput
    ):
        result = OcrMultiOutput.merge([ocrmultioutput_1, ocrmultioutput_empty])

        assert len(result.ocrsos) == 2
        assert result.ocrsos[0].text == "Hello"
        assert result.ocrsos[1].text == "World"

    def test_merge_all_empty_outputs(self, ocrmultioutput_empty: OcrMultiOutput):
        result = OcrMultiOutput.merge([ocrmultioutput_empty, ocrmultioutput_empty])

        assert len(result.ocrsos) == 0

    def test_merge_multiple_outputs(
        self, ocrmultioutput_1: OcrMultiOutput, ocrmultioutput_2: OcrMultiOutput
    ):
        ocrmultioutput_3 = OcrMultiOutput(
            ocrsos=[
                OcrSingleOutput(
                    bbox=Rectangle([[8, 8], [9, 8], [9, 9], [8, 9]]),
                    text="Three",
                    confidence=0.75,
                )
            ]
        )
        result = OcrMultiOutput.merge(
            [ocrmultioutput_1, ocrmultioutput_2, ocrmultioutput_3]
        )

        assert len(result.ocrsos) == 5
        assert result.ocrsos[4].text == "Three"

    def test_merge_preserves_bbox_and_confidence(
        self, ocrmultioutput_1: OcrMultiOutput, ocrmultioutput_2: OcrMultiOutput
    ):
        result = OcrMultiOutput.merge([ocrmultioutput_1, ocrmultioutput_2])

        assert result.ocrsos[0].confidence == 0.9
        assert result.ocrsos[1].confidence == 0.8
        assert result.ocrsos[2].confidence == 0.7
        assert result.ocrsos[3].confidence == 0.85
        assert result.ocrsos[0].bbox is not None
        assert result.ocrsos[1].bbox is not None
        assert result.ocrsos[2].bbox is not None
        assert result.ocrsos[3].bbox is not None

    def test_merge_empty_list(self):
        result = OcrMultiOutput.merge([])
        assert len(result.ocrsos) == 0

    def test_merge_with_none_text_and_confidence(self):
        ocrmultioutput_1 = OcrMultiOutput(
            ocrsos=[
                OcrSingleOutput(
                    bbox=Rectangle([[0, 0], [1, 0], [1, 1], [0, 1]]),
                    text=None,
                    confidence=None,
                )
            ]
        )
        ocrmultioutput_2 = OcrMultiOutput(
            ocrsos=[
                OcrSingleOutput(
                    bbox=Rectangle([[2, 2], [3, 2], [3, 3], [2, 3]]),
                    text="Valid",
                    confidence=0.8,
                )
            ]
        )
        result = OcrMultiOutput.merge([ocrmultioutput_1, ocrmultioutput_2])

        assert len(result.ocrsos) == 2
        assert result.ocrsos[0].text is None
        assert result.ocrsos[0].confidence is None
        assert result.ocrsos[1].text == "Valid"
        assert result.ocrsos[1].confidence == 0.8


class TestOCRMOClosestWord:

    def test_closest_word_returns_nearest1_right(
        self, ocrmultioutput_from_example1: OcrMultiOutput
    ):
        word = ocrmultioutput_from_example1.ocrsos[0]
        assert word.text == "This"
        result = ocrmultioutput_from_example1.closest_word(word, dist_thresh=3)
        assert result is not None
        assert result.text == "is"

    def test_closest_word_returns_nearest1_left(
        self, ocrmultioutput_from_example1: OcrMultiOutput
    ):
        word = ocrmultioutput_from_example1.ocrsos[1]
        assert word.text == "is"
        result = ocrmultioutput_from_example1.closest_word(
            word, dist_thresh=3, _to="left"
        )
        assert result is not None
        assert result.text == "This"

    def test_closest_word_returns_left_empty(
        self, ocrmultioutput_from_example1: OcrMultiOutput
    ):
        word = ocrmultioutput_from_example1.ocrsos[0]
        assert word.text == "This"
        result = ocrmultioutput_from_example1.closest_word(
            word, dist_thresh=3, _to="left"
        )
        assert result is None

    def test_closest_word_returns_right_empty(
        self, ocrmultioutput_from_example1: OcrMultiOutput
    ):
        word = ocrmultioutput_from_example1.ocrsos[5]
        assert word.text == "document."
        result = ocrmultioutput_from_example1.closest_word(
            word, dist_thresh=3, _to="right"
        )
        assert result is None

    def test_closest_word_invalid_direction(
        self, ocrmultioutput_from_example1: OcrMultiOutput
    ):
        word = OcrSingleOutput(
            bbox=Rectangle([[0, 0], [1, 0], [1, 1], [0, 1]]), text=None, confidence=None
        )

        with pytest.raises(ValueError):
            ocrmultioutput_from_example1.closest_word(
                word, dist_thresh=3, _to="invalid"
            )


class TestOCRMOGroupWords:

    def test_group_words(self, ocrmultioutput_from_example1: OcrMultiOutput):
        groupwords, _ = ocrmultioutput_from_example1.group_words(dist_thresh=5)
        assert len(groupwords) == 2
        assert groupwords.ocrsos[0].text == "This is a test PDF document."
        assert (
            groupwords.ocrsos[1].text
            == "If you can read this, you have Adobe Acrobat Reader installed on your computer."
        )

    def test_group_words_symbol_spliter(
        self, ocrmultioutput_from_example1: OcrMultiOutput
    ):
        groupwords, _ = ocrmultioutput_from_example1.group_words(
            dist_thresh=5, symbol_splitter=","
        )
        assert len(groupwords) == 3
        assert groupwords.ocrsos[0].text == "This is a test PDF document."
        assert groupwords.ocrsos[1].text == "If you can read this,"
        assert (
            groupwords.ocrsos[2].text
            == "you have Adobe Acrobat Reader installed on your computer."
        )

    def test_group_words_limit_nmax(self, ocrmultioutput_from_example1: OcrMultiOutput):
        groupwords, _ = ocrmultioutput_from_example1.group_words(
            dist_thresh=5, max_n_words=10
        )
        assert len(groupwords) == 1
        assert groupwords.ocrsos[0].text == "This is a test PDF document."

    def test_group_words_limit_nmin(self, ocrmultioutput_from_example1: OcrMultiOutput):
        groupwords, _ = ocrmultioutput_from_example1.group_words(
            dist_thresh=5, min_n_words=10
        )
        assert len(groupwords) == 1
        assert (
            groupwords.ocrsos[0].text
            == "If you can read this, you have Adobe Acrobat Reader installed on your computer."
        )
