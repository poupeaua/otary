# Linear Entities Processing

Suppose you have a workflow with OCR, a Segment Detection pipeline, a Corner Detection
pipeline, Spline versus Segment classification pipeline for example.

Otary allows you to instantiate linear entities representing the objects you are
manipulating such as Points, Segments, Splines and much more.

You can then display them on the image to visualize the objects you are
manipulating in your workflow.

## Basic Example

```python
from otary import (
    Image,
    Segment,
    Polygon,
    Rectangle,
    VectorizedLinearSpline,
    OcrMultiOutput,
    SegmentsRender,
    LinearSplinesRender,
    OcrSingleOutputRender,
    PolygonsRender,
    PointsRender,
)
from otary.geometry.discrete.linear.directed.entity import DirectedLinearEntity

points = ... # instantiate your contour list of points
dle: list[DirectedLinearEntity] = ... # instantiate you Directed Linear Entities (DLE)
ocrmo: OcrMultiOutput = ... # OCR Multi Output

colors = [
    interpolate_color(i / len(dle)) for i in range(len(dle))
]

for linear_entity, color in zip(dle, colors):
    if isinstance(linear_entity, VectorizedLinearSpline):
        im.draw_splines(
            splines=[linear_entity],
            render=LinearSplinesRender(
                thickness=5,
                default_color=color,
                as_vectors=True,
                pct_ix_head=0.25,
            ),
        )
    elif isinstance(linear_entity, Segment):
        im.draw_segments(
            segments=[linear_entity],
            render=SegmentsRender(
                thickness=5, default_color=color, as_vectors=True
            ),
        )
    else:
        raise RuntimeError(f"Unknown type {type(linear_entity)}")

im.draw_ocr_outputs(
    ocr_outputs=ocrmo.ocrsos,
    render=OcrSingleOutputRender(thickness=2, colors=colors),
)
im.draw_points(
    points=points,
    render=PointsRender(thickness=10, default_color=(0, 0, 0)),
)
```

This previous code would display something like the following image:

![linear-entities](../img/learn/example-otary-linear-entities.png)

## Advanced Manipulation

### Relative Positioning

If you have a similar use-case where you need to create geometrical objects next to pre-existing geometrical objects on the image, you can get inspired by this section.

In the previous example, we displayed not only the OCR boxes but also boxes when
on the symbol "angle" on the left of some OCR boxes. Here is the section of code you would need to add:

```python
for is_angled, ocrso, color in zip(
    self.ocrmo_angle_flags, self.ocrmo.ocrsos, colors
):
    if is_angled:
        shift = -ocrso.bbox.get_vector_left_from_topleft(0).normalized * 40
        segment_left = Segment(
            [
                ocrso.bbox[0],
                ocrso.bbox.get_vertice_from_topleft(0, "bottomleft"),
            ]
        )
        segment_right = segment_left.copy().shift(shift)
        square = Rectangle(
            [
                segment_right[0],
                segment_left[0],
                segment_left[1],
                segment_right[1],
            ],
            regularity_rtol=0.1,
        )
        im.draw_polygons(
            polygons=[square],
            render=PolygonsRender(thickness=2, default_color=color),
        )
```

### Score Geometry entities

Otary allows you to score the confidence of your detected Linear Entities.

You may have detected you Segments or a Contour using OpenCV. Now, you need to
evaluate the quality of the detected objects. Otary allows you to compare the
Linear Entities to the pixels of the image (ground truth).

```python
im_other = im.copy().as_white()

for linear_entity in dle:
    if isinstance(linear_entity, VectorizedLinearSpline):
        im_other.draw_splines(
            splines=[linear_entity],
            render=LinearSplinesRender(
                thickness=5,
                default_color="black",
                as_vectors=False,
                pct_ix_head=0.25,
            ),
        )
    elif isinstance(linear_entity, Segment):
        im_other.draw_segments(
            segments=[linear_entity],
            render=SegmentsRender(
                thickness=5, default_color="black", as_vectors=False
            ),
        )
    else:
        raise RuntimeError(f"Unknown type {type(linear_entity)}")
        
score = im.score_contains_v2(im_other) # compare two images

print(score) # 0.97
```

If you prefer to have a list of scores for each detected Linear Entity, you can
use `score_contains_linear_entities` method instead.

```python
scores = im.score_contains_linear_entities(entities=linear_entities)

print(scores) # [0.96, 0.98, 0.94, 0.89, 0.99, 0.76, 0.82]
```

You can even be more tolerant about the detected objects by dilating the pixels
of the ground truth image. This can be controled by the dilate_kernel and
the dilate_iterations parameters. This way if does not fit exactly but is still close
enough the detected geometry object can be considered as valid for a threshold that you
may choose.

```python
scores = im.score_contains_linear_entities(
    entities=linear_entities,
    dilate_kernel=(5, 5),
    dilate_iterations=2,
)

print(scores) # [0.99, 1.0, 0.99, 0.98, 1.0, 0.95, 0.97]
```

Explore the [Analysis](/api/image/analysis/scoring/#otary.image.image.Image.score_contains_v2) part of the Otary Image module.
With Otary, you can compute the confidence of all your detected Geometry objects or
use it to compare two images.