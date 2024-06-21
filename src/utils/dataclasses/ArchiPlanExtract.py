"""
Image Extract used to gather information about the final information we want to
have for a architectural plan
"""

from dataclasses import dataclass

import numpy as np

from src.geometry import Contour
from src.utils.dataclasses.VectorExtract import VectorExtract


@dataclass
class ArchiPlanExtract:
    """Class for keeping track of a architectural plan"""

    image: np.ndarray
    cnt: Contour
    vextracts: list[VectorExtract]
