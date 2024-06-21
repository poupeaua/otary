"""
Vctor Extract used to gather information about the final information we want to have
for each line
"""

from __future__ import annotations

from dataclasses import dataclass

import src.geometry as geo
from src.utils.dataclasses.OcrSingleOutput import OcrSingleOutput


@dataclass
class VectorExtract:
    """Class for keeping track of a vector extracted information from an image"""

    vector: geo.Vector
    ocrso: OcrSingleOutput

    @property
    def dist_scale_estimate(self):
        return float(self.ocrso.text) / self.vector.length

    @staticmethod
    def mean_dist_scale_estimate(vextracts: list[VectorExtract]) -> tuple[float, float]:
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
        mdse_value = 0
        n_dist_scale_estimates = 0
        for vextract in vextracts:
            if vextract.ocrso.succeeded:
                mdse_value += vextract.dist_scale_estimate
                n_dist_scale_estimates += 1
        mdse_value = mdse_value / n_dist_scale_estimates
        mdse_confidence = n_dist_scale_estimates / len(vextracts)
        return mdse_value, mdse_confidence

    @staticmethod
    def fillna(
        vextracts: list[VectorExtract], ndecimals: int = 0
    ) -> list[VectorExtract]:
        """Fill the OcrSingleOutput that are None, which means that no text has been
        detected, with the estimated mean distance scale.

        Args:
            vextracts (list[VectorExtract]): list of VectorExtract

        Returns:
            list[VectorExtract]: the same list of VectorExtract without None values
                in the ocrso attributes.
        """
        mdse_value, mdse_confidence = VectorExtract.mean_dist_scale_estimate(vextracts)
        for i, vextract in enumerate(vextracts):
            if not vextract.ocrso.succeeded:
                vextracts[i].ocrso = OcrSingleOutput(
                    bbox=None,
                    text=str(round(mdse_value * vextract.vector.length, ndecimals)),
                    confidence=mdse_confidence,
                )
        return vextracts
