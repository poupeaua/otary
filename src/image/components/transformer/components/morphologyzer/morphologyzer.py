from typing import Literal, get_args

import cv2
import numpy as np

from src.image.base import BaseImage

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
    ) -> None:
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

        Returns:
            Self: image object
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
            from src.image import Image

            return Image(image=result)

        self.base.asarray = result

    def resize(
        self, factor: float, interpolation: int = cv2.INTER_AREA, copy: bool = False
    ) -> None:
        """Resize the image to a new size using a scaling factor value that
        will be applied to all dimensions (width and height).

        Applying this method can not result in a distorted image.

        Args:
            factor (float): factor in [0, 5] to resize the image.
                A value of 1 does not change the image.
                A value of 2 doubles the image size.
                A maximum value of 5 is set to avoid accidentally producing a gigantic
                image.

        Returns:
            (Self): resized image
        """
        if factor == 1:
            return self

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

        self.resize_fixed(dim=dim, interpolation=interpolation, copy=copy)

    def blur(
        self,
        kernel: tuple = (5, 5),
        iterations: int = 1,
        method: BlurMethods = "average",
        sigmax: float = 0,
    ) -> None:
        """Blur the images

        Args:
            kernel (tuple, optional): blur kernel size. Defaults to (5, 5).
            iterations (int, optional): number of iterations. Defaults to 1.
            method (str, optional): blur method.
                Must be in ["average", "median", "gaussian", "bilateral"].
                Defaults to "average".

        Returns:
            Self: the new image blurred
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

        Returns:
            Self: image dilated
        """
        if iterations == 0:
            return self

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

        Returns:
            Self: image eroded
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
