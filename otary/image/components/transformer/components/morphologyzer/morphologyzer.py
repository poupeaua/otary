"""
Morphologyzer Transformer component
"""

from __future__ import annotations

from typing import Optional, Literal, get_args, TYPE_CHECKING

import cv2
import numpy as np

from otary.image.base import BaseImage

if TYPE_CHECKING:
    from otary.image import Image

BlurMethods = Literal["average", "median", "gaussian", "bilateral"]


class MorphologyzerImage:
    """MorphologyzerImage."""

    def __init__(self, base: BaseImage) -> None:
        self.base = base

    def resize_fixed(
        self,
        dim: tuple[int, int],
        interpolation: int = cv2.INTER_AREA,
        copy: bool = False,
    ) -> Optional[Image]:
        """Resize the image using a fixed dimension well defined.
        This function can result in a distorted image if the ratio between
        width and height is different in the original and the new image.

        If the dim argument has a negative value in height or width, then
        a proportional ratio is applied based on the one of the two dimension given.

        Args:
            dim (tuple[int, int]): a tuple with two integers in the following order
                (width, height).
            interpolation (int, optional): resize interpolation.
                Defaults to cv2.INTER_AREA.
            copy (bool, optional): whether to return a new image or not.
        """
        if dim[0] < 0 and dim[1] < 0:  # check that the dim is positive
            raise ValueError(f"The dim argument {dim} has two negative values.")

        _dim = list(dim)

        # compute width or height if needed
        if _dim[1] <= 0:
            _dim[1] = int(self.base.height * (_dim[0] / self.base.width))
        if dim[0] <= 0:
            _dim[0] = int(self.base.width * (_dim[1] / self.base.height))

        result = cv2.resize(
            src=self.base.asarray, dsize=_dim, interpolation=interpolation
        )

        if copy:
            from otary.image import Image

            return Image(image=result)

        self.base.asarray = result
        return None

    def resize(
        self, factor: float, interpolation: int = cv2.INTER_AREA, copy: bool = False
    ) -> Optional[Image]:
        """Resize the image to a new size using a scaling factor value that
        will be applied to all dimensions (width and height).

        Applying this method can not result in a distorted image.

        Args:
            factor (float): factor in [0, 5] to resize the image.
                A value of 1 does not change the image.
                A value of 2 doubles the image size.
                A maximum value of 5 is set to avoid accidentally producing a gigantic
                image.
            interpolation (int, optional): resize interpolation.
                Defaults to cv2.INTER_AREA.
            copy (bool, optional): whether to return a new image or not.
        """
        if factor == 1:
            return None

        if factor < 0:
            raise ValueError(
                f"The resize factor value {factor} must be stricly positive"
            )

        max_scale_pct = 5
        if factor > max_scale_pct:
            raise ValueError(f"The resize factor value {factor} is probably too big")

        width = int(self.base.width * factor)
        height = int(self.base.height * factor)
        dim = (width, height)

        return self.resize_fixed(dim=dim, interpolation=interpolation, copy=copy)

    def blur(
        self,
        kernel: tuple = (5, 5),
        iterations: int = 1,
        method: BlurMethods = "average",
        sigmax: float = 0,
    ) -> None:
        """Blur the image

        Args:
            kernel (tuple, optional): blur kernel size. Defaults to (5, 5).
            iterations (int, optional): number of iterations. Defaults to 1.
            method (str, optional): blur method.
                Must be in ["average", "median", "gaussian", "bilateral"].
                Defaults to "average".
            sigmax (float, optional): sigmaX value for the gaussian blur.
                Defaults to 0.
        """
        if method not in list(get_args(BlurMethods)):
            raise ValueError(f"Invalid blur method {method}. Must be in {BlurMethods}")

        for _ in range(iterations):
            if method == "average":
                self.base.asarray = cv2.blur(src=self.base.asarray, ksize=kernel)
            elif method == "median":
                self.base.asarray = cv2.medianBlur(
                    src=self.base.asarray, ksize=kernel[0]
                )
            elif method == "gaussian":
                self.base.asarray = cv2.GaussianBlur(
                    src=self.base.asarray, ksize=kernel, sigmaX=sigmax
                )
            elif method == "bilateral":
                self.base.asarray = cv2.bilateralFilter(
                    src=self.base.asarray, d=kernel[0], sigmaColor=75, sigmaSpace=75
                )

    def dilate(
        self,
        kernel: tuple = (5, 5),
        iterations: int = 1,
        dilate_black_pixels: bool = True,
    ) -> None:
        """Dilate the image by making the black pixels expand in the image.
        The dilatation can be parametrize thanks to the kernel and iterations
        arguments.

        Args:
            kernel (tuple, optional): kernel to dilate. Defaults to (5, 5).
            iterations (int, optional): number of dilatation iterations. Defaults to 1.
            dilate_black_pixels (bool, optional): whether to dilate black pixels or not
        """
        if iterations == 0:
            return None

        if dilate_black_pixels:
            self.base.asarray = 255 - np.asarray(
                cv2.dilate(
                    self.base.rev().asarray,
                    kernel=np.ones(kernel, np.uint8),
                    iterations=iterations,
                ),
                dtype=np.uint8,
            )
        else:  # dilate white pixels by default
            self.base.asarray = np.asarray(
                cv2.dilate(
                    self.base.asarray,
                    kernel=np.ones(kernel, np.uint8),
                    iterations=iterations,
                ),
                dtype=np.uint8,
            )

        return None

    def erode(
        self,
        kernel: tuple = (5, 5),
        iterations: int = 1,
        erode_black_pixels: bool = True,
    ) -> None:
        """Erode the image by making the black pixels shrink in the image.
        The anti-dilatation can be parametrize thanks to the kernel and iterations
        arguments.

        Args:
            kernel (tuple, optional): kernel to erode. Defaults to (5, 5).
            iterations (int, optional): number of iterations. Defaults to 1.
            erode_black_pixels (bool, optional): whether to erode black pixels or not
        """
        if iterations == 0:
            pass

        if erode_black_pixels:
            self.base.asarray = 255 - np.asarray(
                cv2.erode(
                    self.base.rev().asarray,
                    kernel=np.ones(kernel, np.uint8),
                    iterations=iterations,
                ),
                dtype=np.uint8,
            )
        else:
            self.base.asarray = np.asarray(
                cv2.erode(
                    self.base.asarray,
                    kernel=np.ones(kernel, np.uint8),
                    iterations=iterations,
                ),
                dtype=np.uint8,
            )

    def add_border(self, size: int, fill_value: int = 0) -> None:
        """Add a border to the image.

        Args:
            thickness (int): border thickness.
            color (int, optional): border color. Defaults to 0.
        """
        size = int(size)
        self.base.asarray = cv2.copyMakeBorder(
            src=self.base.asarray,
            top=size,
            bottom=size,
            left=size,
            right=size,
            borderType=cv2.BORDER_CONSTANT,
            value=fill_value,
        )  # type: ignore[call-overload]
