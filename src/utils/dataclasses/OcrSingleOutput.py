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
    
    def convert_easyocr(ocr_output: list) -> list[OcrSingleOutput]:
        tmp_list: list[OcrSingleOutput] = []
        for element in ocr_output:
            ocr = OcrSingleOutput(
                bbox=geo.Rectangle(element[0]),
                text=element[1],
                confidence=element[2]
            )
            tmp_list.append(ocr)
        return tmp_list