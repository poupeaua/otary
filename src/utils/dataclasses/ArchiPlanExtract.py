"""
Image Extract used to gather information about the final information we want to have for a
architectural plan
"""

from dataclasses import dataclass
import src
from src.utils.image import Image
from src.utils.dataclasses.SegmentExtract import SegmentExtract

@dataclass
class ArchiPlanExtract:
    """Class for keeping track of a architectural plan"""
    image: Image
    segments_extracts: list[SegmentExtract]