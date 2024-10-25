"""
Utily module to read files (.pdf) 
"""

import io
from typing import Optional
import numpy as np
import pymupdf
from pymupdf import Page
from PIL import Image

def read_pdf_to_images(
        filepath_or_stream: str | io.BytesIO, 
        resolution: Optional[int] = 3508
    ) -> list[np.ndarray]:
    """Read a pdf and turn it into a list of images in a given image resolution.

    Args:
        filepath_or_stream (str | io.BytesIO): filepath or stream of the pdf file.
        resolution (Optional[int], optional): resolution for the output images.
            Defaults to 3508.

    Returns:
        list[np.ndarray]: list of numpy array representing each page as an image.
    """
    if isinstance(filepath_or_stream, io.BytesIO):
        pages = pymupdf.open(stream=filepath_or_stream, filetype="pdf")
    else:
        pages = pymupdf.open(filename=filepath_or_stream)

    images: list[np.ndarray] = []
    for page in pages:
        # computing the rendering for the current page
        if resolution is not None:
            factor = resolution / max(page.rect[-2], page.rect[-1])
            rendering = page.get_pixmap(alpha=False, matrix=pymupdf.Matrix(factor, factor))
        else:
            rendering = page.get_pixmap(alpha=False)
        
        # getting the array
        array = np.array(Image.open(io.BytesIO(rendering.pil_tobytes(format="PNG"))))

        if not array.dtype.type is np.uint8:
            raise TypeError("The array has not the expected type ")
        
        images.append(array)

    return images