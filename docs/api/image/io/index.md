# I/O

The `io` module provides ways to read and write image data. Reading and writing in
Otary is very simple:

```python
import otary as ot

im = ot.Image.from_file(filepath="path/to/file/image")

# any image manipulation you can think of
# ...

im.save(save_filepath="path/to/file/image")
```

## Components

- **[Reader](reader.md):** Responsible for reading image data from various sources.
- **[Writer](writer.md):** Allows you to write image data to various destinations.
