"""
Axis Aligned Rectangle python file
"""

from numpy.typing import NDArray

from otary.geometry.discrete.shape.rectangle import Rectangle

class AxisAlignedRectangle(Rectangle):
    """
    Axis Aligned Rectangle class that inherits from Rectangle.
    It defines a rectangle that is axis-aligned, meaning its sides are parallel 
    to the X and Y axes.
    """

    def __init__(self, points: NDArray | list, is_cast_int: bool = False) -> None:
        if not self.is_axis_aligned:
            raise ValueError("The rectangle is not axis aligned")
        super().__init__(points=points, is_cast_int=is_cast_int)
