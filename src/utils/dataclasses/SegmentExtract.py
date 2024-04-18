"""
Segment Extract used to gather information about the final information we want to have
for each line
"""

from dataclasses import dataclass
import src
from src.geometry import Segment, Rectangle

@dataclass
class SegmentExtract:
    """Class for keeping track of a segment extracted information from an image"""
    segment: Segment
    bbox: Rectangle
    text: str
    confidence: float