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

## Examples

### OCR Outputs Visualization

```python
import otary as ot

im = ot.Image.from_file(filepath="path/to/file/image")

ocr_output = pytesseract.image_to_data(
    im.as_pil(),
    output_type=pytesseract.Output.DICT
)

ocrmo = OcrMultiOutput.from_pytesseract(ocr_output)

# display image with OCR outputs
im.draw_ocr_outputs(
    ocr_output=ocrmo.ocrsos, # OcrMultiOutput to list[OcrSingleOutput]
    render=ot.OcrSingleOutputRender()
)

im.show()
```

### Retrieve a Key Value pair from an OCR Output

Imagine you have an OCR output, and you want to extract a specific key-value pair from it. You have for example a document with several key-value pairs, and you want to extract the value associated with a specific key.

For example you could have a `Name:` word and next to it, on the right, a given text
which represents the value (name).
You can make it automatically using the `HeuristicKeyInformationExtractor`.

```python
import otary as ot

im = im.Image.from_file(filepath="path/to/file/image").as_grayscale()

reader = easyocr.Reader(['en'])
ocr_output = reader.readtext(im.asarray)

ocrmo = ot.OcrMultiOutput.from_easyocr(ocr_output)
value = ot.HeuristicKeyInformationExtractor.extract(
    ocr_outputs=ocrmo,
    key="your_key",
    closest_word_dist_thresh=im.dist_pct(pct=0.3),
)
```

Some other Deep Learning models are just more efficient for KIE tasks (see DONUT model). However, they are harder to fine-tune, maintain, etc... This approach provides a good
balance between performance and ease of use.

It can especially useful to start a new project, get quicly started, and you want to fine-tune later. Or if you do not have enough training data.
