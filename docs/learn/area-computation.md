# Area Computation

## Compute Intersection Over Union and more

![area-computation](../img/learn/example-otary-img-geometry.png)

``` py linenums="1"
import otary as ot

im = ot.Image.from_file("../tests/data/vision/example2/sample-otary-img1.pdf")


polygon_array = np.array(
    [[60, 200], [130, 130], [270, 130.1], [270, 300], [250, 500], [60, 500]],
    dtype=np.float32
)

polygon = ot.Polygon(polygon_array)
aabb = polygon.aabb().expand(1.1)

print(aabb.iou(polygon)) # 0.7789 (IOU = Intersection Over Union)
print(aabb.inter_area(polygon) / polygon.area) # 1.0
print(aabb.union_area(polygon) / aabb.area) # 1.0
```
