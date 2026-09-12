# Image

The `image` module provides a flexible and powerful way to work with images.
Otary aims to make image processing easy and accessible to everyone.

Here is a sample python code to show what Otary can do:

``` py linenums="1"
import otary as ot

im = ot.Image.from_file(filepath="path/to/file/image")

im.crop(x0=50, y0=50, x1=450, y1=450)
im.rotate(angle=90, is_degree=True)

im.save(save_filepath="path/to/file/image")
im.show()
```

## Image Channel Order

Otary follows OpenCV's BGR channel convention for 3-channel images. Make sure image arrays are in BGR order when working with Otary.

If the image is already in RGB order, pass `is_bgr=False` when calling output methods such as `show()` or `save()`.

```python
import otary as ot

im = ot.Image(rgb_array)

im.show(is_bgr=False)

im.save("output.png", is_bgr=False)
```

## Components

The `image` module is built around the following components:

- **[I/O](io/index.md):** Responsible for reading and writing image data.
- **[Analysis](analysis/index.md):** Perform image analysis tasks
- **[Transformers](transformers/index.md):** Allows you to apply various transformations to the image, such as resizing, cropping, and color adjustments
- **[Drawer](drawer/index.md):** Provides methods for drawing shapes and text on the image.

Behind the scene the components all use a base image class.
