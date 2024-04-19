"""
Vctor Extract used to gather information about the final information we want to have
for each line
"""

from dataclasses import dataclass
import src
import src.geometry as geo

@dataclass
class VectorExtract:
    """Class for keeping track of a vector extracted information from an image"""
    vector: geo.Vector
    bbox: geo.Rectangle
    text: str
    confidence: float