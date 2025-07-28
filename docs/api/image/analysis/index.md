# Analyzer

The `analyzer` module provides a set of functions for analyzing images.

For example, if you want to know whether a given segment is present in the image you
can use the `score_contains_segments` function:

```python
import otary as ot

im = ot.Image.from_file(filepath="path/to/file/image")

segment = ot.Segment([[100, 100], [200, 200]])

score = im.score_contains_segments(segments=[segment]) # score is between 0 and 1
```

## Components

- **[Scoring](scoring.md):** scoring operations such as IoU (Intersection over Union).
    Those score methods are special in the sense that the output is always a float
    number between 0 and 1
