"""
Utility module to read files (.pdf)
"""

import io
from typing import Optional
import numpy as np
import pymupdf
from PIL import Image as ImagePIL


def read_pdf_document(
    filepath_or_stream: str | io.BytesIO, page_nb: Optional[int] = None
) -> list[pymupdf.Page]:
    """Read pdf into a list of PyMuPDF Page

    Args:
        filepath_or_stream (str | io.BytesIO): filepath or stream of the pdf file.
        page_nb (int, optional): page to load. Default to None which indicates that
            we load all pages from the pdf.

    Returns:
        list[pymupdf.Page]: list of PyMuPDF Page object
    """
    if isinstance(filepath_or_stream, io.BytesIO):
        pages = pymupdf.open(stream=filepath_or_stream, filetype="pdf")
    else:
        valid_format = ["pdf"]
        file_format = filepath_or_stream.split(".")[-1]
        if file_format not in valid_format:
            raise ValueError(f"The filepath is not in any valid format {valid_format}")
        pages = pymupdf.open(filename=filepath_or_stream)

    # set pages if page_nb
    if page_nb is not None:
        if page_nb < 0:
            page_nb = len(pages) + page_nb
        try:
            pages = [pages[page_nb]]
        except IndexError as exc:
            raise IndexError(
                f"The page number {page_nb} is not correct as the pdf contains "
                f"{len(pages)} page(s)."
            ) from exc

    return pages


def read_pdf_to_images(
    filepath_or_stream: str | io.BytesIO,
    resolution: Optional[int] = 3508,
    page_nb: Optional[int] = None,
    clip_pct: Optional[pymupdf.Rect] = None,
) -> list[np.ndarray]:
    """Read a pdf and turn it into a list of images in a given image resolution.

    Args:
        filepath_or_stream (str | io.BytesIO): filepath or stream of the pdf file.
        resolution (Optional[int], optional): resolution for the output images.
            Defaults to 3508.
        page_nb (int, optional): page to load. Default to None which indicates that
            we load all pages from the pdf.
        clip_pct (pymmupdf.Rect, optional): optional zone to extract in the image.
            This is particularly useful to load into memory only a small part of the
            image without loading everything into memory. This reduces considerably
            the image loading time especially combined with a high resolution.

    Returns:
        list[np.ndarray]: list of numpy array representing each page as an image.
    """
    pages = read_pdf_document(filepath_or_stream=filepath_or_stream, page_nb=page_nb)

    if clip_pct is not None:
        page_rect = pages[0].bound()
        clip = pymupdf.Rect(
            x0=clip_pct.x0 * page_rect.width,
            y0=clip_pct.y0 * page_rect.height,
            x1=clip_pct.x1 * page_rect.width,
            y1=clip_pct.y1 * page_rect.height,
        )
    else:
        clip = None

    images: list[np.ndarray] = []
    for page in pages:
        if resolution is not None:
            factor = resolution / max(page.rect[-2], page.rect[-1])
            rendering = page.get_pixmap(
                alpha=False, matrix=pymupdf.Matrix(factor, factor), clip=clip
            )
        else:
            rendering = page.get_pixmap(alpha=False, clip=clip)

        array = np.array(ImagePIL.open(io.BytesIO(rendering.pil_tobytes(format="PNG"))))

        if array.dtype.type is not np.uint8:
            raise TypeError("The array has not the expected type ")

        images.append(array)

    return images
