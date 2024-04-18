"""
Line Extract used to gather information about the final information we want to have
for each line
"""

from dataclasses import dataclass
import src
from src.utils.image import Image

@dataclass
class UrbanDocExtract:
    """Class for keeping track of a Urban Document extracted information"""
    image: Image
    surface: float
    scale: str
    creation_date: str
    drawer: str
    register: str
    email: str
    ciudad: str
    locality: str
    use: str
    type: str
    