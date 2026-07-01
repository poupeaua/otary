# API

Otary is built upon two core modules **Image** and **Geometry**, each designed with a distinct architectural pattern to provide flexibility and power.

## [Image: A Composition-Based Approach](image/index.md)

The `image` module is designed following a **composition over inheritance** principle. This allows you to build complex image processing pipelines by combining smaller, independent, and reusable components. Instead of inheriting properties from a monolithic class, you can dynamically assemble functionality, leading to a more flexible and maintainable codebase.

The image object is composed of a `Reader` to load data, `Writer` to save data, a `Transformer` to apply modifications and a `Drawer` to add overlays, all working
together seamlessly.

## [Geometry: An Inheritance-Based Structure](geometry/index.md)

The `geometry` module uses a more traditional **inheritance** model. It provides a clear and hierarchical structure for geometric entities. Base classes define common behaviors, and specialized subclasses inherit and extend this functionality. This approach is ideal for creating a well-defined and logical classification of shapes and geometric objects.

## [Vision: Computer Vision tools](vision/index.md)

The `vision` module provides tools for manipulating OCR outputs. Otary does not include
any OCR engine, but it provides an interface for working with OCR outputs.

You can use this module to process and analyze OCR outputs extracted from your
favorite OCR engines, such as Tesseract, EasyOCR, DocTR, Azure Document Intelligence, Amazon Textract, etc...

This way you can easily display OCR outputs on top of your images, analyze your OCR outputs, extract information using a heuristic Key Information or key-value pairs retrieval method, and much more.
