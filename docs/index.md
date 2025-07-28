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
<a href="https://github.com/poupeaua/otary/tree/master?tab=GPL-3.0-1-ov-file" target="_blank"><img src="https://img.shields.io/github/license/poupeaua/otary?color=8A2BE2&label=license" alt="License"></a>
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
2. draw an ellipse on it
3. crop a part of the image
4. rotate the cropped image
5. apply a threshold
6. show the image

In order to compare the use of Otary versus other libraries, I will use the same example but with different libraries. Try it yourself on your favorite LLM (like [ChatGPT](https://chatgpt.com/)) by copying the query:

```text
Generate a python code to read an image from a pdf, draw an ellipse on it, crop a part of the image, rotate the cropped image, apply a threshold on the image.
```

Using Otary you can do it with few lines of code:

=== "Otary"

    ```Python
    import otary as ot

    im = ot.Image.from_pdf("path/to/your/file.pdf", page_nb=0)

    ellipse = ot.Ellipse(foci1=[100, 100], foci2=[400, 400], semi_major_axis=250)

    im = (
        im.draw_ellipses([ellipse])
        .crop(x0=50, y0=50, x1=450, y1=450)
        .rotate(angle=90, is_degree=True)
        .threshold_simple(thresh=200)
    )

    im.show()
    ```

=== "Other libraries"

    ```Python
    """
    Providing the input to ChatGPT gives the following code
    """
    import fitz  # PyMuPDF
    import numpy as np
    import cv2

    def read_image_from_pdf(pdf_path, page_number=0, dpi=300):
        """Extracts the specified page as an image from a PDF."""
        doc = fitz.open(pdf_path)
        page = doc[page_number]
        mat = fitz.Matrix(dpi / 72, dpi / 72)  # scale to DPI
        pix = page.get_pixmap(matrix=mat)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    def draw_ellipse(img, center, axes, angle=0, color=(0, 255, 0), thickness=2):
        """Draws an ellipse on the image."""
        return cv2.ellipse(img.copy(), center, axes, angle, 0, 360, color, thickness)

    def crop_image(img, top_left, bottom_right):
        """Crops the image using top-left and bottom-right coordinates."""
        x1, y1 = top_left
        x2, y2 = bottom_right
        return img[y1:y2, x1:x2]

    def rotate_image(img, angle):
        """Rotates the image around its center by the given angle."""
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return rotated

    def apply_threshold(img, thresh_value=127):
        """Applies a binary threshold on the grayscale version of the image."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_BINARY)
        return thresh

    def main():
        pdf_path = "your_file.pdf"

        # Step 1: Read image from PDF
        img = read_image_from_pdf(pdf_path)

        # Step 2: Draw an ellipse on the image
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        axes = (w // 4, h // 6)
        img_with_ellipse = draw_ellipse(img, center, axes, angle=30, color=(0, 0, 255), thickness=3)

        # Step 3: Crop a part of the image
        cropped_img = crop_image(img_with_ellipse, (100, 100), (500, 500))

        # Step 4: Rotate the cropped image
        rotated_img = rotate_image(cropped_img, angle=45)

        # Step 5: Apply threshold
        thresholded_img = apply_threshold(rotated_img, thresh_value=150)

        # Display results
        cv2.imshow("Ellipse Image", img_with_ellipse)
        cv2.imshow("Cropped Image", cropped_img)
        cv2.imshow("Rotated Image", rotated_img)
        cv2.imshow("Thresholded Image", thresholded_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Optionally save results
        cv2.imwrite("ellipse_image.jpg", img_with_ellipse)
        cv2.imwrite("cropped_image.jpg", cropped_img)
        cv2.imwrite("rotated_image.jpg", rotated_img)
        cv2.imwrite("thresholded_image.jpg", thresholded_img)

    if __name__ == "__main__":
        main()
    ```

ChatGPT proposes to re-invent the wheel.

Using Otary makes the code:

- Much more **readable** and hence maintainable
- Much more **interactive**
- Much simpler, simplifying **libraries management** by only using one library and not manipulating multiple libraries like Pillow, OpenCV, Scikit-Image, PyMuPDF etc.

## Enhanced Interactivity

!!! tip "Enhanced Interactivity"

    In a Jupyter notebook, you can easily test and iterate on transformations by simply commenting part of the code as you need it.

    ```python
    im = (
        im.draw_ellipses([ellipse])
        # .crop(x0=50, y0=50, x1=450, y1=450)
        # .rotate(angle=90, is_degree=True)
        .threshold_simple(thresh=200)
    )
    ```
