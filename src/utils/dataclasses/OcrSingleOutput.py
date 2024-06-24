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

    @property
    def succeeded(self) -> bool:
        """Check if an OCR extraction has been possible

        Returns:
            bool: if True, and OCR extraction has been possible and thus we have
                attached to this object a valid bounding box. If False, no OCR
                extraction was possible and no bouding box is available.
        """
        return self.bbox is not None

    def is_complete(self, with_bbox: bool = True) -> bool:
        """Check whether this OcrSingleOutput object is complete or not.
        Complete means that it is valid with no inappropriate values whatsoever.

        Args:
            with_bbox (bool, optional): Whether to consider the bounding boxes or not.
                If False, we just verify if the text and confidence are ok.
                Defaults to False.

        Returns:
            bool: if True the MultiVectorExtract object is complete without any
                inappropriate values.
        """
        cond_text = (
            isinstance(self.text, str) and self.text != "" and self.text is not None
        )
        cond_conf = (
            isinstance(self.confidence, float)
            and self.confidence >= 0
            and self.confidence <= 1
        )
        cond_bbox = True

        if with_bbox:
            cond_bbox = self.succeeded and isinstance(self.bbox, geo.Rectangle)

        return cond_text and cond_conf and cond_bbox

    @staticmethod
    def convert_easyocr(ocr_output: list) -> list[OcrSingleOutput]:
        """Convert an easyocr formatted output from the read method into a common
        OcrSingleOutput format. The idea is to make the code agnostic of the OCR used.
        The OCR could be EasyOCR, Tesserocr, KerasOCR, etc... this would not affect the
        program.

        Args:
            ocr_output (list): In easyocr package currently the output is a list of
                tuples that contains in this order the bounding box, the text and
                the confidence about the text read.

        Returns:
            list[OcrSingleOutput]: list of OcrSingleOutput objects
        """
        ocrsos: list[OcrSingleOutput] = [
            OcrSingleOutput(bbox=geo.Rectangle(e[0]), text=e[1], confidence=e[2])
            for e in ocr_output
        ]
        return ocrsos

    def __str__(self) -> str:
        return (
            f"OcrSingleOutput(bbox={str(self.bbox)}, text={self.text}, "
            f"confidence={self.confidence})"
        )

    def __repr__(self) -> str:
        return (
            f"OcrSingleOutput(bbox={str(self.bbox)}, text={self.text}, "
            f"confidence={self.confidence})"
        )
