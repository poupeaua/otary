"""
conftest.py is a special type of test file that makes it
possible to use automatically (without explicitly importing) the fixtures
in all the subdirectories.
"""

import json
import pytest

from otary.vision.ocr.ocr_multi_output import OcrMultiOutput


@pytest.fixture
def ocrmultioutput_from_example1() -> OcrMultiOutput:
    """Document"""
    example1_easyocr_output = json.load(
        open("tests/data/vision/example1/output_easyocr.json", "r")
    )
    return OcrMultiOutput.from_easyocr(example1_easyocr_output)
