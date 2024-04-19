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
    _id: str
    surface: float
    scale: str
    creation_date: str
    drawer_name: str
    drawer_register: str
    drawer_email: str
    region: str 
    ciudad: str # municipio
    locality: str # localidad
    use: str
    type: str
    