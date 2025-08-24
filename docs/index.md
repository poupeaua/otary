<p align="center">
  <a href="">
    <img src="https://github.com/poupeaua/otary/raw/master/docs/img/logo-withname-bg-transparent.png" alt="Otary">
</a>
</p>

<p align="center">
    <em>Otary library, shape your images, image your shapes.</em>
</p>

<p align="center">
<a href="https://alexandrepoupeau.com/otary/" > <img src="https://gradgen.bokub.workers.dev/badge/rainbow/Otary%20%20%20?gradient=d76333,edb12f,dfc846,6eb8c9,1c538b&label=Enjoy"/></a>
<a href="https://github.com/poupeaua/otary/actions/workflows/test.yaml" > <img src="https://github.com/poupeaua/otary/actions/workflows/test.yaml/badge.svg"/></a>
<a href="https://codecov.io/github/poupeaua/otary" > <img src="https://codecov.io/github/poupeaua/otary/graph/badge.svg?token=LE040UGFZU"/></a>
<a href="https://app.codacy.com/gh/poupeaua/otary/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade" > <img src="https://app.codacy.com/project/badge/Grade/704a873ee08c40318423a47ec71b9bf4"/></a>
<a href="https://alexandrepoupeau.com/otary/" > <img src="https://github.com/poupeaua/otary/actions/workflows/docs.yaml/badge.svg?branch=master"/></a>
<a href="https://pypi.org/project/otary" target="_blank"> <img src="https://img.shields.io/pypi/v/otary?color=blue&label=pypi" alt="Package version"></a>
<a href="https://pypi.org/project/otary" target="_blank"><img src="https://img.shields.io/pypi/pyversions/otary?color=blue&label=python" alt="License"></a>
<a href="https://github.com/poupeaua/otary/blob/master/LICENSE" target="_blank"><img src="https://img.shields.io/github/license/poupeaua/otary?color=8A2BE2" alt="License"></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
</p>

# Welcome to Otary

Otary — elegant, readable, and powerful image and 2D geometry Python library.

## Features

The main features of Otary are:

- **Unification**: Otary offers a cohesive solution for image and geometry manipulation, letting you work seamlessly without switching tools.

- **Readability**: Self-explanatory by design. Otary’s clean, readable code eliminates the need for comments, making it easy for beginners to learn and for experts to build efficiently.

- **Performance**: optimized for speed and efficiency, making it suitable for high-performance applications. It is built on top of [NumPy](https://numpy.org) and [OpenCV](https://opencv.org), which are known for their speed and performance.

- **Interactivity**: designed to be Interactive and user-friendly, ideal for [Jupyter notebooks](https://jupyter.org) and live exploration.

- **Flexibility**: provides a flexible and extensible architecture, allowing developers to customize and extend its functionality as needed.

## Example

Let me illustrate the usage of Otary with a simple example. Imagine you need to:

1. read an image from a pdf file
2. draw an rectangle on it, shift and rotate the rectangle
3. crop a part of the image
4. rotate the cropped image
5. apply a threshold
6. show the image

In order to compare the use of Otary versus other libraries, I will use the same example but with different libraries. Try it yourself on your favorite LLM (like [ChatGPT](https://chatgpt.com/)) by copying the query:

```text
Generate a python code to read an image from a pdf, draw an rectangle on it, shift and rotate the rectangle, crop a part of the image, rotate the cropped image, apply a threshold on the image.
```

Using Otary you can do it with few lines of code:

=== "Otary"

    ```Python
    import otary as ot

    im = ot.Image.from_pdf("path/to/you/file.pdf", page_nb=0)

    rectangle = ot.Rectangle([[1, 1], [4, 1], [4, 4], [1, 4]]) * 100
    rectangle.shift([50, 50]).rotate(angle=30, is_degree=True)

    im = (
        im.draw_polygons([rectangle])
        .crop(x0=50, y0=50, x1=450, y1=450)
        .rotate(angle=90, is_degree=True)
        .threshold_simple(thresh=200)
    )

    im.show()
    ```

=== "ChatGPT using other libraries"

    ```Python
    #!/usr/bin/env python3
    """
    Steps:
    - Load first page of a PDF as an image
    - Draw a rectangle
    - Shift & rotate that rectangle (visualized as a rotated box)
    - Crop a region of the image
    - Rotate the cropped image
    - Threshold the (rotated) crop

    Dependencies:
        pip install pdf2image Pillow opencv-python
        # If pdf2image isn't available, install: pip install PyMuPDF
        # Note: pdf2image requires Poppler on your system.

    Edit the CONFIG section below to suit your needs.
    """

    from pathlib import Path
    import math

    # Pillow & OpenCV
    from PIL import Image, ImageDraw
    import numpy as np
    import cv2

    # Try to import a PDF rasterizer
    _loader = None
    try:
        from pdf2image import convert_from_path
        _loader = "pdf2image"
    except Exception:
        try:
            import fitz  # PyMuPDF
            _loader = "pymupdf"
        except Exception:
            _loader = None


    # --------------------------- CONFIG --------------------------- #
    PDF_PATH = "example.pdf"        # <- put your PDF path here
    OUTPUT_DIR = Path("out_steps")
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Rectangle (axis-aligned) you want to draw first:
    rect_x, rect_y, rect_w, rect_h = 200, 150, 400, 250  # pixels

    # Shift to apply to the rectangle center (dx, dy):
    shift_dx, shift_dy = 120, -40  # pixels

    # Rotation to apply to the rectangle (degrees, positive=CCW):
    rotate_deg = 25.0

    # Crop region from the original image (x, y, w, h):
    crop_x, crop_y, crop_w, crop_h = 100, 100, 600, 400

    # Rotation to apply to the cropped image (degrees):
    crop_rotate_deg = -15.0

    # Threshold (use None to use Otsu automatically)
    fixed_threshold_value = None  # e.g., set to 128 to force a fixed threshold
    # -------------------------------------------------------------- #


    def load_pdf_first_page_as_image(pdf_path: str, dpi: int = 300) -> Image.Image:
        """Return the first page of a PDF as a Pillow RGB image."""
        if _loader == "pdf2image":
            pil_pages = convert_from_path(pdf_path, dpi=dpi, first_page=1, last_page=1)
            if not pil_pages:
                raise RuntimeError("No pages found in PDF.")
            return pil_pages[0].convert("RGB")
        elif _loader == "pymupdf":
            doc = fitz.open(pdf_path)
            if doc.page_count == 0:
                raise RuntimeError("No pages found in PDF.")
            page = doc.load_page(0)
            # 300 dpi equivalent scaling
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            return img
        else:
            raise ImportError(
                "No PDF rasterizer available. Install either `pdf2image` (plus Poppler) or `PyMuPDF`."
            )


    def pil_to_cv(img_pil: Image.Image) -> np.ndarray:
        """Pillow RGB -> OpenCV BGR"""
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


    def cv_to_pil(img_cv: np.ndarray) -> Image.Image:
        """OpenCV BGR -> Pillow RGB"""
        return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))


    def draw_axis_aligned_rectangle_pil(img_pil: Image.Image, x, y, w, h, width=4):
        """Draw axis-aligned rectangle on a PIL image."""
        draw = ImageDraw.Draw(img_pil)
        draw.rectangle([x, y, x + w, y + h], outline=(255, 0, 0), width=width)
        return img_pil


    def draw_rotated_rectangle_cv(img_cv: np.ndarray, center, size, angle_deg: float, thickness=3, color=(0, 255, 0)):
        """
        Draw a rotated rectangle using OpenCV. center=(cx,cy), size=(w,h), angle in degrees CCW.
        """
        rect = (center, size, angle_deg)
        box = cv2.boxPoints(rect)  # 4x2 float32 array of vertices
        box = np.int32(box)
        cv2.polylines(img_cv, [box], isClosed=True, color=color, thickness=thickness)
        return img_cv


    def rotate_image_keep_bounds(img_cv: np.ndarray, angle_deg: float) -> np.ndarray:
        """
        Rotate an image about its center, expanding bounds so nothing is cropped.
        """
        (h, w) = img_cv.shape[:2]
        c = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(c, angle_deg, 1.0)
        # compute new bounds
        cos = abs(M[0, 0])
        sin = abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        # adjust rotation matrix to account for translation
        M[0, 2] += (new_w / 2) - c[0]
        M[1, 2] += (new_h / 2) - c[1]
        rotated = cv2.warpAffine(img_cv, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return rotated


    def threshold_image(img_cv_gray: np.ndarray, fixed_thresh: int | None = None) -> np.ndarray:
        """
        Apply binary threshold. If fixed_thresh is None, use Otsu.
        """
        if fixed_thresh is None:
            _, th = cv2.threshold(img_cv_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        else:
            _, th = cv2.threshold(img_cv_gray, int(fixed_thresh), 255, cv2.THRESH_BINARY)
        return th


    def main():
        # 1) Load first page
        pil_img = load_pdf_first_page_as_image(PDF_PATH, dpi=300)
        pil_img.save(OUTPUT_DIR / "01_loaded_page.png")

        # 2) Draw axis-aligned rectangle (Pillow)
        pil_with_rect = pil_img.copy()
        pil_with_rect = draw_axis_aligned_rectangle_pil(pil_with_rect, rect_x, rect_y, rect_w, rect_h, width=4)
        pil_with_rect.save(OUTPUT_DIR / "02_axis_aligned_rect.png")

        # Convert to OpenCV for further operations
        cv_img = pil_to_cv(pil_with_rect)

        # 3) Shift & rotate rectangle (OpenCV rotated box)
        #    Start from the original rectangle center:
        cx = rect_x + rect_w / 2.0
        cy = rect_y + rect_h / 2.0
        #    Apply shift
        cx_shifted = cx + shift_dx
        cy_shifted = cy + shift_dy
        #    Draw rotated rectangle (in green)
        cv_img_rotrect = cv_img.copy()
        cv_img_rotrect = draw_rotated_rectangle_cv(
            cv_img_rotrect,
            center=(cx_shifted, cy_shifted),
            size=(rect_w, rect_h),
            angle_deg=rotate_deg,
            thickness=3,
            color=(0, 255, 0),
        )
        cv2.imwrite(str(OUTPUT_DIR / "03_shifted_rotated_rect.png"), cv_img_rotrect)

        # 4) Crop a region (axis-aligned box on the original image)
        x1, y1 = int(crop_x), int(crop_y)
        x2, y2 = int(crop_x + crop_w), int(crop_y + crop_h)
        h, w = cv_img.shape[:2]
        # clamp to image
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w, x2))
        y2 = max(0, min(h, y2))
        crop = cv_img[y1:y2, x1:x2].copy()
        cv2.imwrite(str(OUTPUT_DIR / "04_crop.png"), crop)

        # 5) Rotate the cropped image (keeping bounds)
        crop_rot = rotate_image_keep_bounds(crop, crop_rotate_deg)
        cv2.imwrite(str(OUTPUT_DIR / "05_crop_rotated.png"), crop_rot)

        # 6) Threshold the (rotated) crop
        crop_gray = cv2.cvtColor(crop_rot, cv2.COLOR_BGR2GRAY)
        crop_th = threshold_image(crop_gray, fixed_threshold_value)
        cv2.imwrite(str(OUTPUT_DIR / "06_crop_threshold.png"), crop_th)

        print("Done. See outputs in:", OUTPUT_DIR.resolve())


    if __name__ == "__main__":
        main()
    ```

ChatGPT proposes to re-invent the wheel and over-complicates everything.

Using Otary makes the code:

- Much more **readable** and hence **maintainable**
- Much more **interactive**
- Much simpler, simplifying **libraries management** by only using one library and not manipulating multiple libraries like Pillow, OpenCV, Scikit-Image, PyMuPDF etc.

## Enhanced Interactivity

!!! tip "Enhanced Interactivity"

    In a Jupyter notebook, you can easily test and iterate on transformations by simply commenting part of the code as you need it.

    ```python
    im = (
        im.draw_polygons([rectangle])
        # .crop(x0=50, y0=50, x1=450, y1=450)
        # .rotate(angle=90, is_degree=True)
        .threshold_simple(thresh=200)
    )
    ```
