# Drawer

Drawing methods can be accessed through the `drawer` attribute of the image object.

However all the methods of the drawer are also available as direct methods of the image object to provide a more user friendly API.
For example, drawing a circle in the image is as easy as:

```python
import otary as ot

im = ot.Image.from_file(filepath="path/to/file/image")

circle = ot.Circle(center=[100, 100], radius=50)

im.draw_circles(
    circles=[circle],
    render=ot.CirclesRender(thickness=5, default_color="red")
)
```

## Components

The drawer module is divided into two parts:

- **[Drawing](drawing.md):** provides methods to draw on images
- **[Renderers](renderers.md):**  used to define the style of the objects to be drawn
