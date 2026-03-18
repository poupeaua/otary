"""
OCR Output used to gather information about the output of any OCR model
"""

from __future__ import annotations

import re
from typing import Optional

import numpy as np

import otary.geometry as geo


class OcrSingleOutput:
    """Class for keeping track of ONLY ONE OCR extracted information from an image"""

    def __init__(
        self,
        bbox: geo.Rectangle,
        text: Optional[str] = None,
        confidence: Optional[float] = None,
        objectness: Optional[float] = None,
    ) -> None:
        """Initialize an OcrSingleOutput object

        Args:
            bbox (Rectangle): bounding boxes
            text (Optional[str], optional): text extracted by OCR within the bounding
                box. It can be None if no text was extracted. Imagine only using this
                after a simple text detection model. No text recognition model.
                In that case, we would have only the bounding box and no text.
                Defaults to None.
            confidence (Optional[float], optional): text confidence score.
                Defaults to None.
            objectness (Optional[float], optional): bounding box objectness score.
                which can be seen as the bounding box confidence score.
                Defaults to None.
        """
        self.bbox = bbox
        self.text = text  # type: ignore
        self.confidence = confidence  # type: ignore
        self.objectness = objectness  # type: ignore

    @property
    def succeeded(self) -> bool:
        """Check if an OCR extraction has been possible

        Returns:
            bool: if True, and OCR extraction has been possible and thus we have
                attached to this object a valid bounding box. If False, no OCR
                extraction was possible and no bounding box is available.
        """
        return self.bbox is not None

    def contains_string(self, string: list[str] | str, cond: str = "any") -> bool:
        """Check if the OCR single output contains a given string or list of strings.

        Args:
            string (list[str] | str): list of strings to be found in the OCR single
                output text. Can be a single string or a list of strings.
            cond (str, optional): condition. Must be in [any, all].
                If any, the method returns true if any of the strings is in the OCR
                single output text. If all, returns true only if all strings values are
                found in OCR single output text. Defaults to "any".

        Returns:
            bool: True if text contains strings.
        """
        if self.text is None:
            return False

        if isinstance(string, str):
            string = [string]

        tests = []
        for txt in string:
            if txt in self.text:
                tests.append(1)
            else:
                tests.append(0)

        if cond == "any":
            return bool(np.any(tests))
        if cond == "all":
            return bool(np.all(tests))
        raise ValueError(f"The parameter cond must be in [any, all]. Found {cond}")

    def contains_regex(self, regex: str) -> bool:
        """Check if OCR single output text contains the regex pattern.

        Args:
            regex (str): regex to search in the OCR single output text.

        Returns:
            bool: True or False
        """
        if self.text is None:
            return False

        if re.search(pattern=regex, string=self.text) is not None:
            return True
        return False

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

    def copy(self) -> OcrSingleOutput:
        """Return a copy of the OcrSingleOutput object.

        Returns:
            OcrSingleOutput: a copy of the current OcrSingleOutput object.
        """
        return OcrSingleOutput(
            bbox=self.bbox.copy(),
            text=self.text,
            confidence=self.confidence,
            objectness=self.objectness,
        )

    def __str__(self) -> str:
        return (
            f"OcrSingleOutput(bbox={str(self.bbox)}, text='{self.text}', "
            f"confidence={self.confidence})"
        )

    def __repr__(self) -> str:
        return (
            f"OcrSingleOutput(bbox={str(self.bbox)}, text='{self.text}', "
            f"confidence={self.confidence})"
        )
