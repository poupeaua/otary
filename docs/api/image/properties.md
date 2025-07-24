# Properties

Properties are the essential attributes of an image.
They provide basic information about the image, such as its dimensions, shape, etc.

Here is for example a python sample code to get the width of the image:

```python
import otary as ot

im = ot.Image.from_fillvalue(shape=(256, 128, 3), value=255)

print(im.width) # 128
```

::: otary.image.image.Image
    options:
        show_root_heading: false
        show_root_toc_entry: false
        show_source: true
        members:
            - asarray
            - asarray_binary
            - width
            - height
            - center
            - area
            - channels
            - shape_array
            - shape
            - is_gray
            - norm_side_length
            - corners
