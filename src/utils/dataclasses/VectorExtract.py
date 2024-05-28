"""
Vctor Extract used to gather information about the final information we want to have
for each line
"""

from dataclasses import dataclass

import src.geometry as geo
from src.utils.dataclasses.OcrSingleOutput import OcrSingleOutput


@dataclass
class VectorExtract:
    """Class for keeping track of a vector extracted information from an image"""

    vector: geo.Vector
    ocr: OcrSingleOutput
    dist_scale_estimate: float
