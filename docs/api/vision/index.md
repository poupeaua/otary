# Vision

The `vision` module provides tools manipulate and analyze OCR outputs.
Otary's vision layer is focused on OCR outputs and key information extraction, offering a common abstraction over multiple OCR engines and heuristics for structured text discovery.

The module is built around `OcrSingleOutput` and `OcrMultiOutput`, which represent detected text, bounding boxes, confidence scores, and objectness information. It includes adapter methods for converting outputs from engines such as Tesseract, EasyOCR, DocTR, Azure Document Intelligence, AWS Textract, etc... into a unified format.

Key information extraction utilities help locate values associated with expected labels (for example invoice numbers, dates, or totals) using fuzzy matching and spatial relationships.

## Key Components

- **OCR Outputs:** [OcrSingleOutput](ocr/ocr_single_output.md), [OcrMultiOutput](ocr/ocr_multi_output.md)
    - **OCR Engine Adapters:**
        - [from_pytesseract](ocr/ocr_multi_output.md/#otary.vision.ocr.ocr_multi_output.OcrMultiOutput.from_pytesseract)
        - [from_easyocr](ocr/ocr_multi_output.md/#otary.vision.ocr.ocr_multi_output.OcrMultiOutput.from_easyocr)
        - [from_doctr](ocr/ocr_multi_output.md/#otary.vision.ocr.ocr_multi_output.OcrMultiOutput.from_doctr)
        - [from_azure_document_intelligence](ocr/ocr_multi_output.md/#otary.vision.ocr.ocr_multi_output.OcrMultiOutput.from_azure_document_intelligence)
        - [from_aws_textract](ocr/ocr_multi_output.md/#otary.vision.ocr.ocr_multi_output.OcrMultiOutput.from_aws_textract)
- **Key Information Extraction:** [HeuristicKeyInformationExtractor](kie/heuristic.md)

## How to use

You can jump directly in the [Learn section about OCR](../../learn/ocr.md) to see more.
