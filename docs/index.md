<p align="center">
  <a href="">
    <img src="https://github.com/poupeaua/otary/raw/master/docs/img/logo-withname-bg-transparent.png" alt="Otary">
</a>
</p>

<p align="center">
    <em>Otary library, readable, fast, unified, interactive, flexible, pythonic</em>
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

Otary is a powerful Python library for advanced image and 2D geometry manipulation.

## Features

The main features of Otary are:

- **Readability**: designed to be easy to read and understand, making it suitable for beginners and experienced developers alike.

- **Performance**: optimized for speed and efficiency, making it suitable for high-performance applications. It is built on top of [NumPy](https://numpy.org) and [OpenCV](https://opencv.org), which are known for their speed and performance.

- **Unification**: Otary unifies multiple libraries into a single, unified library, making it easier to use without the need to switch between multiple libraries. Spend less time learning different APIs and reading multiple documentations.

- **Interactiveness**: designed to be interactive and user-friendly, making it suitable for interactive applications like Jupyter notebooks.

- **Flexibility**: provides a flexible and extensible architecture, allowing developers to customize and extend its functionality as needed.

- **Pythonic**: designed to be Pythonic and easy to use, making it suitable for Python developers.

## Example

Let me illustrate the usage of Otary with a simple example. Imagine you need to:

1. read an image from a pdf file
2. crop a part of it
3. rotate the cropped image
4. apply a threshold
5. draw a ellipse on it
6. show the image

Try it out yourself on your favorite LLM (like [ChatGPT](https://chatgpt.com/)) by copying the query:

```text
Read an image from a pdf, crop a part of it given by a topleft point plus the width and the height of crop bbox, then rotate the cropped image, apply a threshold on the image. Finally draw a ellipse on it and show the image.
```

=== "Otary"

    ```Python
    import otary as ot

    im = ot.Image.from_pdf(filepath="path/to/you/file.pdf", page_nb=0)

    ellipse = ot.Ellipse(foci1=[10, 10], foci2=[50, 50], semi_major_axis=50)

    im = (
        im.crop_from_topleft(topleft=[200, 100], width=100, height=100)
        .rotate(angle=90, is_degree=True, is_clockwise=False)
        .threshold_simple(thresh=200)
        .draw_ellipses(
            ellipses=[ellipse],
            render=ot.EllipsesRender(
                is_draw_focis_enabled=True,
                default_color="red"
            )
        )
    )

    im.show()
    ```

=== "Other libraries"

    ```Python
    import fitz  # PyMuPDF
    import cv2
    import numpy as np

    def process_pdf_crop_rotate_threshold_draw(
        pdf_path,
        page_number,
        topleft_x,
        topleft_y,
        width,
        height,
        rotation=cv2.ROTATE_90_CLOCKWISE
    ):
        # Load PDF and render page
        doc = fitz.open(pdf_path)
        page = doc[page_number]
        pix = page.get_pixmap(dpi=300)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Crop ROI
        cropped = gray[topleft_y:topleft_y+height, topleft_x:topleft_x+width]

        # Rotate cropped image
        rotated = cv2.rotate(cropped, rotation)

        # Apply threshold (Otsu)
        _, thresh = cv2.threshold(rotated, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Draw ellipse on the thresholded image
        center = (thresh.shape[1] // 2, thresh.shape[0] // 2)
        axes = (thresh.shape[1] // 4, thresh.shape[0] // 4)
        cv2.ellipse(thresh, center, axes, angle=0, startAngle=0, endAngle=360, color=128, thickness=2)

        # Show the final image
        cv2.imshow("Processed Image", thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Optionally save
        cv2.imwrite("final_output.png", thresh)

        return thresh

    # Example usage:
    if __name__ == "__main__":
        process_pdf_crop_rotate_threshold_draw(
            pdf_path="example.pdf",
            page_number=0,
            topleft_x=100,
            topleft_y=200,
            width=300,
            height=400
        )
    ```

- Otary makes the code much more **readable**
- Otary makes the code much more **interactive**
- Otary makes **libraries management easier** by only using one library and not depending on multiple libraries like Pillow, OpenCV, Scikit-Image, PyMuPDF etc.

!!! tip "Enhanced Interactiveness"

    In a Jupyter notebook, you can easily test and iterate on transformations by simply commenting part of the code as you need it.

    ```python
    im = (
        im.crop_from_topleft(topleft=[200, 100], width=100, height=100)
        # .rotate(angle=90, is_degree=True, is_clockwise=False)
        # .threshold_simple(thresh=200)
        .draw_ellipses(
            ellipses=[ellipse],
            render=ot.EllipsesRender(is_draw_focis_enabled=True)
        )
    )
    ```
