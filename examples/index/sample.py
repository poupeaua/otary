import otary as ot

im = ot.Image.from_pdf("tests/data/test.pdf", page_nb=0)

rectangle = ot.Rectangle([[1, 1], [4, 1], [4, 4], [1, 4]]) * 100
rectangle.shift([50, 50]).rotate(angle=30, is_degree=True)

im = (
    im.draw_polygons([rectangle])
    .crop(x0=50, y0=50, x1=450, y1=450)
    .rotate(angle=90, is_degree=True)
    .threshold_simple(thresh=200)
)

im.show()
