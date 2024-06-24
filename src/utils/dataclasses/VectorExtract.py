"""
Vctor Extract used to gather information about the final information we want to have
for each line
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import src.geometry as geo
from src.utils.dataclasses.OcrSingleOutput import OcrSingleOutput
from src.text.norm import normalize_str_number, is_str_number


@dataclass
class VectorExtract:
    """Class for keeping track of a vector extracted information from an image"""

    vector: geo.Vector
    ocrso: OcrSingleOutput

    @property
    def scale_estimate(self) -> float:
        """Compute the scale estimation based on ocr extraction and length of the
        vector.

        Returns:
            float: distance scale estimation
        """
        txt = normalize_str_number(self.ocrso.text)
        if not is_str_number(txt):
            raise RuntimeError(
                f"The text {self.ocrso.text} can not be transformed" f"into a number"
            )
        return float(txt) / self.vector.length

    def __str__(self) -> str:
        return f"VectorExtract(vector={str(self.vector)}, orcso={str(self.ocrso)})"

    def __repr__(self) -> str:
        return f"VectorExtract(vector={str(self.vector)}, orcso={str(self.ocrso)})"


class MultiVectorExtract:
    def __init__(self, vextracts: list[VectorExtract]) -> None:
        self.vextracts = vextracts

    @property
    def is_fully_unsucceeded(self) -> bool:
        """Whether the scale estimate is possible to compute or not.

        Returns:
            bool: True if the MultiVectorExtract object contains not a single
                OCR extraction, False otherwise.
        """
        return self.n_succeeded == 0

    @property
    def is_fully_succeeded(self) -> bool:
        """Whether there is an OCR extraction for every VectorExtract

        Returns:
            bool: True if all VectorExtract have a valid OCR extraction,
                False otherwise.
        """
        return self.n_succeeded == len(self)

    @property
    def n_succeeded(self) -> int:
        """Number of valid OCR extraction

        Returns:
            int: Number of valid OCR extraction
        """
        return sum([v.ocrso.succeeded for v in self.vextracts])

    @property
    def confidence_scale_estimate(self) -> float:
        """The confidence given to the scale estimate

        Returns:
            float: a number in [0, 1] confidence of the distance scale estimate.
        """
        return self.n_succeeded / len(self)

    @property
    def mean_scale_estimate(self) -> float:
        """Compute the mean distance scale estimate of the contour.
        It provides the confidence about this estimate too. It is done by dividing
        the number of distance scale estimates that could be computed from the number
        of VectorExtracts (number of segments in the contour).

        Args:
            vextracts (list[VectorExtract]): list of VectorExtract

        Returns:
            tuple[float, float]: the mean distance scale estimate and the confidence
                about this number.
        """
        assert not self.is_fully_unsucceeded
        mse_value = 0
        for vextract in self.vextracts:
            if vextract.ocrso.succeeded:
                mse_value += vextract.scale_estimate
        mse_value = mse_value / self.n_succeeded
        return mse_value

    def fillna(self, ndecimals: int = 0) -> MultiVectorExtract:
        """Fill the OcrSingleOutput that are None, which means that no text has been
        detected, with the estimated mean distance scale.

        Args:
            ndecimals (int): number of decimals accepted for scale estimation production

        Returns:
            MultiVectorExtract: the same list of VectorExtract without None values
                in the ocrso attributes.
        """
        if self.is_fully_succeeded:
            return self
        if self.is_fully_unsucceeded:
            raise ValueError(
                "There is not a single succeeded OCR extraction."
                "The mean distance scale estimation can not be computed."
            )

        new_vextracts = deepcopy(self.vextracts)
        for i, vextract in enumerate(self.vextracts):
            if not vextract.ocrso.succeeded:
                new_vextracts[i].ocrso = OcrSingleOutput(
                    bbox=None,
                    text=str(
                        round(
                            self.mean_scale_estimate * vextract.vector.length, ndecimals
                        )
                    ),
                    confidence=self.confidence_scale_estimate,
                )
        return MultiVectorExtract(new_vextracts)

    def is_complete(self, with_bbox: bool = True) -> bool:
        """Whether the MultiVectorExtract is complete and ready to be used for
        transcription.

        Args:
            with_bbox (bool, optional): Whether to consider the bounding boxes or not.
                If False, we just verify if the text and confidence are ok.
                Defaults to False.

        Returns:
            bool: if True the MultiVectorExtract object is complete without any
                inappropriate values.
        """
        for vextract in self.vextracts:
            if not vextract.ocrso.is_complete(with_bbox=with_bbox):
                return False
        return True

    def __len__(self):
        """Number of elements in the vextracts attribute"""
        return len(self.vextracts)

    def __str__(self) -> str:
        string = "MultiVectorExtract("
        for vextract in self.vextracts:
            string += f"\n{vextract}, "
        string = string[:-2] + "\n)"
        return string

    def __repr__(self) -> str:
        return self.__str__()
