"""
OCR Output used to gather information about the output of any OCR model
"""

from __future__ import annotations

from dataclasses import dataclass

import src.geometry as geo


@dataclass
class OcrSingleOutput:
    """Class for keeping track of a OCR extracted information from an image"""

    bbox: geo.Rectangle
    text: str
    confidence: float

    @staticmethod
    def convert_easyocr(ocr_output: list) -> list[OcrSingleOutput]:
        """Convert an easyocr formatted output from the read method into a common
        OcrSingleOutput format. The idea is to make the code agnostic of the OCR used.

        Args:
            ocr_output (list): In easyocr package currently the output is a list of
                tuples that contains in this order the bounding box, the text read and
                the confidence about the text read.

        Returns:
            list[OcrSingleOutput]: list of OcrSingleOutput objects
        """
        tmp_list: list[OcrSingleOutput] = []
        for element in ocr_output:
            ocr = OcrSingleOutput(
                bbox=geo.Rectangle(element[0]), text=element[1], confidence=element[2]
            )
            tmp_list.append(ocr)
        return tmp_list
