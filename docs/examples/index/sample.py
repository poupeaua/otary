import otary as ot

im = ot.Image.from_pdf("tests/data/test.pdf", page_nb=0)

ellipse = ot.Ellipse(foci1=[100, 100], foci2=[400, 400], semi_major_axis=250)

im = (
    im.draw_ellipses([ellipse])
    .crop(x0=50, y0=50, x1=450, y1=450)
    .rotate(angle=90, is_degree=True)
    .threshold_simple(thresh=200)
)

im.show()
