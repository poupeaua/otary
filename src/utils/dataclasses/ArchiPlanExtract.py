"""
Image Extract used to gather information about the final information we want to
have for a architectural plan
"""

from dataclasses import dataclass

import numpy as np

from src.utils.dataclasses.VectorExtract import VectorExtract


@dataclass
class ArchiPlanExtract:
    """Class for keeping track of a architectural plan"""

    image: np.ndarray
    surface: float
    perimeter: float
    estimate_distance_scale: float
    shape_category: str  # regular or irregular
    contour_extract: list[
        VectorExtract
    ]  # TODO verify is contour and first line with point most at north-east
