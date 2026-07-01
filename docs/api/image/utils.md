# Utils Methods

Conversion is the process of converting an image to something else, which can be anything.

For example, here is an example to convert an image to grayscale:

```python
import otary as ot

im = ot.Image.from_file(filepath="path/to/file/image")

im.dict_pct(0.01) # represent 1% of the image diagonal length
```

::: otary.image.image.Image
    options:
        show_root_heading: false
        show_root_toc_entry: false
        show_source: true
        members:
            - copy
            - dist_pct
            - is_equal_shape
            - rev
