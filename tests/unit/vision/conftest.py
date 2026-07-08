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


@pytest.fixture
def azure_output() -> dict:
    return json.load(open("tests/data/vision/example2/output_azure.json", "r"))


@pytest.fixture
def textract_output() -> dict:
    return {
        "Blocks": [
            {
                "Geometry": {
                    "BoundingBox": {
                        "Width": 0.053907789289951324,
                        "Top": 0.08913730084896088,
                        "Left": 0.11085548996925354,
                        "Height": 0.013171200640499592,
                    },
                    "Polygon": [
                        {"Y": 0.08985357731580734, "X": 0.11085548996925354},
                        {"Y": 0.08913730084896088, "X": 0.16447919607162476},
                        {"Y": 0.10159222036600113, "X": 0.16476328670978546},
                        {"Y": 0.10230850428342819, "X": 0.11113958805799484},
                    ],
                },
                "Text": "Hello, world.",
                "TextType": "PRINTED",
                "BlockType": "LINE",
                "Confidence": 99.56285858154297,
                "Id": "d7fbd604-d609-4d69-857d-247a3f591238",
                "Relationships": [
                    {
                        "Type": "CHILD",
                        "Ids": [
                            "7f97e2ca-063e-47a8-981c-8beee31afc01",
                            "4b990aa0-af96-4369-b90f-dbe02538ed21",
                        ],
                    }
                ],
            },
            {
                "Geometry": {
                    "BoundingBox": {
                        "Width": 0.053907789289951324,
                        "Top": 0.08913730084896088,
                        "Left": 0.11085548996925354,
                        "Height": 0.013171200640499592,
                    },
                    "Polygon": [
                        {"Y": 0.08985357731580734, "X": 0.11085548996925354},
                        {"Y": 0.08913730084896088, "X": 0.16447919607162476},
                        {"Y": 0.10159222036600113, "X": 0.16476328670978546},
                        {"Y": 0.10230850428342819, "X": 0.11113958805799484},
                    ],
                },
                "Text": "Hello,",
                "TextType": "PRINTED",
                "BlockType": "WORD",
                "Confidence": 99.74746704101562,
                "Id": "7f97e2ca-063e-47a8-981c-8beee31afc01",
            },
            {
                "Geometry": {
                    "BoundingBox": {
                        "Width": 0.053907789289951324,
                        "Top": 0.08913730084896088,
                        "Left": 0.11085548996925354,
                        "Height": 0.013171200640499592,
                    },
                    "Polygon": [
                        {"Y": 0.08985357731580734, "X": 0.11085548996925354},
                        {"Y": 0.08913730084896088, "X": 0.16447919607162476},
                        {"Y": 0.10159222036600113, "X": 0.16476328670978546},
                        {"Y": 0.10230850428342819, "X": 0.11113958805799484},
                    ],
                },
                "Text": "world.",
                "TextType": "PRINTED",
                "BlockType": "WORD",
                "Confidence": 99.5171127319336,
                "Id": "4b990aa0-af96-4369-b90f-dbe02538ed21",
            },
        ]
    }
