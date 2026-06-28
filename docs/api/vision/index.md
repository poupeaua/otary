# Vision

The `vision` module provides tools for extracting and interpreting text from images.
Otary's vision layer is focused on OCR outputs and key information extraction, offering a common abstraction over multiple OCR engines and heuristics for structured text discovery.

The module is built around `OcrSingleOutput` and `OcrMultiOutput`, which represent detected text, bounding boxes, confidence scores, and objectness information. It includes adapter methods for converting outputs from engines such as Tesseract, EasyOCR, and DocTR into a unified format.

Key information extraction utilities help locate values associated with expected labels (for example invoice numbers, dates, or totals) using fuzzy matching and spatial relationships.

## Key Components

- **OCR Outputs:** [OcrSingleOutput](ocr/ocr_single_output), [OcrMultiOutput](ocr/ocr_multi_output)
    - **OCR Engine Adapters:** [from_pytesseract](ocr/ocr_multi_output/#otary.vision.ocr.ocr_multi_output.OcrMultiOutput.from_pytesseract), [from_easyocr](ocr/ocr_multi_output/#otary.vision.ocr.ocr_multi_output.OcrMultiOutput.from_easyocr), [from_doctr](ocr/ocr_multi_output/#otary.vision.ocr.ocr_multi_output.OcrMultiOutput.from_doctr)
- **Key Information Extraction:** [HeuristicKeyInformationExtractor](kie/heuristic)

## Example

```python
from otary.vision.ocr import OcrMultiOutput
from otary.vision.kie.heuristic import HeuristicKeyInformationExtractor

ocr_output = OcrMultiOutput.from_easyocr(easyocr_output)
values = HeuristicKeyInformationExtractor.extract(
    ocr_outputs=ocr_output,
    key="your_key",
    closest_word_dist_thresh=50.0
)
```

## Available Modules

Below is a list of available modules and their functionalities:

### Optical Character Recognition (OCR) outputs

::: otary.vision.ocr.ocr_single_output
::: otary.vision.ocr.ocr_multi_output

### Key Information Extraction

::: otary.vision.kie.heuristic
