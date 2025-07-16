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

## Installation

Otary is available on [PyPI](https://pypi.org/project/otary/). You can install it with:

```bash
pip install otary
```

## Example

Let me illustrate the usage of Otary with a simple example. Imagine you need to:

1. read an image from a pdf file
2. crop a part of it
3. rotate the cropped image
4. apply athreshold
5. draw a ellipse on it
6. show the image

Try it out yourself on your favorite LLM (like [ChatGPT](https://chatgpt.com/)) by copying the query:

```text
Read an image from a pdf, crop a part of it given by a topleft point plus the width and the height of crop bounding box, then rotate the cropped image, apply a threshold on the image. Finally draw a ellipse on it and show the image.
```

Using Otary you can do it with few lines of code:


```python
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

- Otary makes the code much more **readable**
- Otary makes the code much more **interactive**
- Otary makes **libraries management easier** by only using one library and not depending on multiple libraries like Pillow, OpenCV, Scikit-Image, PyMuPDF etc.


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
