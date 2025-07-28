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

## Installation

Otary is available on [PyPI](https://pypi.org/project/otary/). You can install it with:

```bash
pip install otary
```

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

```python
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

Using Otary makes the code:

- Much more **readable** and hence maintainable
- Much more **interactive**
- Much simpler, simplifying **libraries management** by only using one library and not manipulating multiple libraries like Pillow, OpenCV, Scikit-Image, PyMuPDF etc.

## Enhanced Interactivity

In a Jupyter notebook, you can easily test and iterate on transformations by simply commenting part of the code as you need it.

```python
im = (
    im.draw_ellipses([ellipse])
    # .crop(x0=50, y0=50, x1=450, y1=450)
    # .rotate(angle=90, is_degree=True)
    .threshold_simple(thresh=200)
)
```
