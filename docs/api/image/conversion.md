# Conversion Methods

Conversion is the process of converting an image to something else, which can be anything.

For example, here is an example to convert an image to grayscale:

```python
import otary as ot

im = ot.Image.from_file(filepath="path/to/file/image")

im.as_grayscale()
```

::: otary.image.image.Image
    options:
        show_root_heading: false
        show_root_toc_entry: false
        show_source: true
        members:
            - as_grayscale
            - as_colorscale
            - as_filled
            - as_white
            - as_black
            - as_pil
            - as_api_file_input
