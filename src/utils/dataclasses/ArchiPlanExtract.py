"""
Image Extract used to gather information about the final information we want to have for a
architectural plan
"""

from dataclasses import dataclass
import src
from src.utils.image import Image
from src.utils.dataclasses import VectorExtract

@dataclass
class ArchiPlanExtract:
    """Class for keeping track of a architectural plan"""
    image: Image
    surface: float
    perimeter: float
    estimate_distance_scale: float
    shape_category: str # regular or irregular
    contour_extract: list[VectorExtract] #TODO verify is contour and first line with point most at north-east