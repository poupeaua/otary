# Why Otary ?

## Unification

**“*One library. Images and geometry together.*”**

Otary is a unified library for image and geometry processing, enabling easy manipulation and interaction between images and geometric objects without the need to use multiple libraries.

If you are tired of switching between [Numpy](https://numpy.org/), [OpenCV](https://opencv.org/), 
[Shapely](https://shapely.readthedocs.io/), [PyMuPDF](https://pymupdf.readthedocs.io/) or fitz,
[Matplotlib](https://matplotlib.org/), [Pillow](https://pillow.readthedocs.io/), 
[Scikit-Image](https://scikit-image.org/), [Sympy](https://sympy.org/), [pdf2image](https://pypi.org/project/pdf2image/), etc... you are not alone.

You can always use the specific libraries if you need a more low-level control over your image processing pipeline.

## Readability

**“*No comment.*”**

Otary is designed so the API speaks for itself. Otary’s ultimate goal is to make you never need to write comments at all.

Like Uncle Bob says in his book *Clean Code*, if you need to write a comment you have failed to express yourself through your code.

Thus one of the main focus of Otary is to make your code as readable and easy to understand as possible.

## Performance

Otary’s image module uses [OpenCV](https://opencv.org) for fast, efficient image processing, while its geometry module relies on [NumPy](https://numpy.org) to deliver high-performance numerical and geometric operations.

Otary is optimized for speed and efficiency, making it suitable for high-performance applications.

## Interactivity

Otary is designed to be Interactive and user-friendly, ideal for [Jupyter notebooks](https://jupyter.org) and live exploration.

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

## Flexibility

Otary provides a flexible and extensible architecture, allowing developers to customize and extend its functionality as needed. 

If you are a Python developer, interested by Otary and want to contribute, feel free to bring your ideas to life! Check the [Contributing](../about/contributing) section for more details.
