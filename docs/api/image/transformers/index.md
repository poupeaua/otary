# Transformers

Transformers can be accessed using the `transformer` attribute of the image object.
From here you can access all the specific transformer attributes: `cropper`, `geometrizer`, `morphologyzer`, `binarizer`.

However to make the developer experience more user friendly, the transformers are also available as direct methods of the image object.
For example, cropping an image is as easy as:

```python
import otary as ot

im = ot.Image.from_file(filepath="path/to/file/image")

im.crop(x0=50, y0=50, x1=450, y1=450)
```

Otary gives you the choice between being more explicit or more synthetic in your code.

## Components

Here are the different specialized components that can be used to transform images:

- **[Cropping](cropping):** cropping methods
- **[Geometry](geometry):** geometric operations
- **[Morphology](morphology):** morphological operations
- **[Binarization & Thresholding](thresholding):** binarization and thresholding
- **[Scoring](scoring):** scoring operations such as iou (intersection over union)

## The `copy` parameter

Some methods (all the cropping methods, some morphology methods like resize, etc...) have a boolean `copy` parameter.

By default, the `copy` parameter is set to `False` which means that the original image is modified and then returned.
This is the default behaviour of all the methods in the Otary library.

However, when `copy` is set to `True`, a new `Image` object is returned and the original image is not modified.
This is useful when you want to create a new image after the transformation without modifying the original image.

Consider an example where you want to crop an image:

!!! failure "Wrong way of cropping and preserving the original image"

    This approach works but can be **considerably slower** especially when the image is large because you first copy the image and then crop it.

    ```python
    import otary as ot

    im = ot.Image.from_file(filepath="path/to/file/image")

    im_crop = im.copy().crop(x0=50, y0=50, x1=450, y1=450)
    ```

This would instead be the correct way to crop and preserve the original image:

!!! success "Good way of cropping and preserving the original image"

    ```python
    import otary as ot

    im = ot.Image.from_file(filepath="path/to/file/image")

    im_crop = im.crop(x0=50, y0=50, x1=450, y1=450, copy=True)
    ```
