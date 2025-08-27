"""
Binarizer component
"""

from typing import Literal, get_args

import cv2
import numpy as np
from numpy.typing import NDArray

from otary.image.base import BaseImage

from otary.image.components.transformer.components.binarizer.utils.thresholding import (
    threshold_isauvola,
    threshold_niblack_like,
    threshold_su,
)

BinarizationMethods = Literal[
    "adaptative", "otsu", "niblack", "sauvola", "wolf", "nick", "isauvola", "wan"
]


class BinarizerImage:
    # pylint: disable=line-too-long
    """BinarizerImage class

    It includes different binarization methods:

    | Name      | Year | Paper                                                                                                                                         |
    |-----------|------|-----------------------------------------------------------------------------------------------------------------------------------------------|
    | Adaptative|  -   | [OpenCV Adaptive Thresholding Documentation](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html)                                |
    | Otsu      | 1979 | [A Threshold Selection Method from Gray-Level Histograms](https://ieeexplore.ieee.org/document/4310076)                                       |
    | Niblack   | 1986 | "An Introduction to Digital Image Processing" by Wayne Niblack                                                                                |
    | Sauvola   | 1997 | [Adaptive Document Binarization](https://www.researchgate.net/publication/3710586_Adaptive_Document_Binarization)                             |
    | Wolf      | 2003 | [Extraction and Recognition of Artificial Text in Multimedia Documents](https://hal.science/hal-01504401v1)                                                                                 |
    | Nick      | 2009 | [Comparison of Niblack inspired Binarization Methods for Ancient Documents](https://www.researchgate.net/publication/221253803_Comparison_of_Niblack_inspired_Binarization_Methods_for_Ancient_Documents) |
    | ISauvola  | 2016 | [ISauvola: Improved Sauvola’s Algorithm for Document Image Binarization](https://www.researchgate.net/publication/304621554_ISauvola_Improved_Sauvola) |
    | Wan       | 2018 | [Binarization of Document Image Using Optimum Threshold Modification](https://www.researchgate.net/publication/326026836_Binarization_of_Document_Image_Using_Optimum_Threshold_Modification) |
    """

    def __init__(self, base: BaseImage) -> None:
        self.base = base

    # ----------------------------- GLOBAL THRESHOLDING -------------------------------

    def threshold_simple(self, thresh: int) -> None:
        """Compute the image thesholded by a single value T.
        All pixels with value v <= T are turned black and those with value v > T are
        turned white. This is a global thresholding method.

        Args:
            thresh (int): value to separate the black from the white pixels.
        """
        self.base.as_grayscale()
        self.base.asarray = np.array((self.base.asarray > thresh) * 255, dtype=np.uint8)

    def threshold_otsu(self) -> None:
        """Apply Otsu global thresholding.
        This is a global thresholding method that automatically determines
        an optimal threshold value from the image histogram.

        Paper (1979):
        https://ieeexplore.ieee.org/document/4310076

        Consider applying a gaussian blur before for better thresholding results.
        See why in https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html.

        As the input image must be a grayscale before applying any thresholding
        methods we convert the image to grayscale.
        """
        self.base.as_grayscale()
        _, img_thresholded = cv2.threshold(
            self.base.asarray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        self.base.asarray = img_thresholded

    # ------------------------------ LOCAL THRESHOLDING -------------------------------

    def threshold_adaptative(self, block_size: int = 11, constant: float = 2.0) -> None:
        """Apply adaptive local thresholding.
        This is a local thresholding method that computes the threshold for a pixel
        based on a small region around it.

        A gaussian blur is applied before for better thresholding results.
        See why in https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html.

        As the input image must be a grayscale before applying any thresholding
        methods we convert the image to grayscale.

        Args:
            block_size (int, optional): Size of a pixel neighborhood that is used to
                calculate a threshold value for the pixel: 3, 5, 7, and so on.
                Defaults to 11.
            constant (int, optional): Constant subtracted from the mean or weighted
                mean. Normally, it is positive but may be zero or negative as well.
                Defaults to 2.
        """
        self.base.as_grayscale()
        self.base.asarray = cv2.adaptiveThreshold(
            src=self.base.asarray,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=block_size,
            C=constant,
        )

    def threshold_niblack(self, window_size: int = 15, k: float = -0.2) -> None:
        """Apply Niblack local thresholding.

        Book (1986):
        "An Introduction to Digital Image Processing" by Wayne Niblack.

        See https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_niblack_sauvola.html

        Args:
            window_size (int, optional): apply on the
                image. Defaults to 15.
            k (float, optional): factor to apply to regulate the impact
                of the std. Defaults to -0.2.
        """
        self.base.as_grayscale()
        self.base.asarray = threshold_niblack_like(
            img=self.base.asarray, method="niblack", window_size=window_size, k=k
        )[1]

    def threshold_sauvola(
        self, window_size: int = 15, k: float = 0.5, r: float = 128.0
    ) -> None:
        """Apply Sauvola local thresholding.
        This is a local thresholding method that computes the threshold for a pixel
        based on a small region around it.

        Paper (1997):
        https://www.researchgate.net/publication/3710586

        See https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_niblack_sauvola.html.

        As the input image must be a grayscale before applying any thresholding
        methods we convert the image to grayscale.

        Args:
            window_size (int, optional): sauvola window size to apply on the
                image. Defaults to 15.
            k (float, optional): sauvola k factor to apply to regulate the impact
                of the std. Defaults to 0.5.
            r (float, optional): sauvola r value. Defaults to 128.
        """
        self.base.as_grayscale()
        self.base.asarray = threshold_niblack_like(
            img=self.base.asarray, method="sauvola", window_size=window_size, k=k, r=r
        )[1]

    def threshold_wolf(self, window_size: int = 15, k: float = 0.5) -> None:
        """Apply Wolf local thresholding.

        Paper (2003):
        https://hal.science/hal-01504401v1

        Args:
            window_size (int, optional): apply on the
                image. Defaults to 15.
            k (float, optional): factor to apply to regulate the impact
                of the std. Defaults to 0.5.
        """
        self.base.as_grayscale()
        self.base.asarray = threshold_niblack_like(
            img=self.base.asarray, method="wolf", window_size=window_size, k=k
        )[1]

    def threshold_nick(self, window_size: int = 19, k: float = -0.1) -> None:
        """Apply Nick local thresholding.

        Paper (2009):
        https://www.researchgate.net/publication/221253803

        The paper suggests to use a window size of 19 and a k factor in [-0.2, -0.1].

        Args:
            window_size (int, optional): apply on the
                image. Defaults to 15.
            k (float, optional): factor to apply to regulate the impact
                of the std. Defaults to -0.1.
        """
        self.base.as_grayscale()
        self.base.asarray = threshold_niblack_like(
            img=self.base.asarray, method="nick", window_size=window_size, k=k
        )[1]

    def threshold_su(
        self,
        window_size: int = 3,
        n_min: int = -1,
    ) -> None:
        """Compute the Su local thresholding.

        Paper (2010):
        https://www.researchgate.net/publication/220933012

        Args:
            window_size (int, optional): window size for high contrast image 
                computation. Defaults to 3.            
            n_min (int, optional): minimum number of high contrast pixels within the 
                neighborhood window. Defaults to -1 meaning that n_min = window_size.
        """
        self.base.as_grayscale()
        self.base.asarray = threshold_su(
            img=self.base.asarray, window_size=window_size, n_min=n_min
        )

    def threshold_isauvola(
        self,
        window_size: int = 15,
        k: float = 0.01,
        r: float = 128.0,
        connectivity: int = 8,
        contrast_window_size: int = 3,
        opening_n_min_pixels: int = 0,
        opening_connectivity: int = 8,
    ) -> None:
        """Apply ISauvola local thresholding.

        Paper (2016):
        https://www.researchgate.net/publication/304621554

        Args:
            window_size (int, optional): apply on the
                image. Defaults to 15.
            k (float, optional): factor to apply to regulate the impact
                of the std. Defaults to 0.01.
            r (float, optional): factor to apply to regulate the impact
                of the std. Defaults to 128.
            connectivity (int, optional): connectivity to apply on the
                image. Defaults to 8.
            contrast_window_size (int, optional): contrast window size to apply on the
                image. Defaults to 3.
            opening_n_min_pixels (int, optional): opening n min pixels to apply on the
                image. Defaults to 0.
            opening_connectivity (int, optional): opening connectivity to apply on the
                image. Defaults to 8.
        """
        # pylint: disable=too-many-arguments, too-many-positional-arguments
        self.base.as_grayscale()
        self.base.asarray = threshold_isauvola(
            img=self.base.asarray,
            window_size=window_size,
            k=k,
            r=r,
            connectivity=connectivity,
            contrast_window_size=contrast_window_size,
            opening_n_min_pixels=opening_n_min_pixels,
            opening_connectivity=opening_connectivity,
        )

    def threshold_wan(
        self, window_size: int = 15, k: float = 0.5, r: float = 128.0
    ) -> None:
        """Apply Wan local thresholding.

        Paper (2018):
        https://www.researchgate.net/publication/326026836

        Args:
            window_size (int, optional): apply on the
                image. Defaults to 15.
            k (float, optional): factor to apply to regulate the impact
                of the std. Defaults to 0.5.
        """
        self.base.as_grayscale()
        self.base.asarray = threshold_niblack_like(
            img=self.base.asarray, method="wan", window_size=window_size, k=k, r=r
        )[1]

    # ---------------------------- BINARY REPRESENTATION ------------------------------

    def binary(self, method: BinarizationMethods = "sauvola") -> NDArray:
        """Binary representation of the image with values that can be only 0 or 1.
        The value 0 is now 0 and value of 255 are now 1. Black is 0 and white is 1.
        We can also talk about the mask of the image to refer to the binary
        representation of it.

        The sauvola is generally the best binarization method however it is
        way slower than the others methods. The adaptative or otsu method are the best
        method in terms of speed and quality.

        Args:
            method (str, optional): the binarization method to apply.
                Must be in ["adaptative", "otsu", "sauvola", "niblack", "nick", "wolf"].
                Defaults to "sauvola".

        Returns:
            NDArray: array where its inner values are 0 or 1
        """
        if method not in list(get_args(BinarizationMethods)):
            raise ValueError(
                f"Invalid binarization method {method}. "
                f"Must be in {BinarizationMethods}"
            )
        getattr(self, f"threshold_{method}")()
        return self.base.asarray_binary

    def binaryrev(self, method: BinarizationMethods = "sauvola") -> NDArray:
        """Reversed binary representation of the image.
        The value 0 is now 1 and value of 255 are now 0. Black is 1 and white is 0.
        This is why it is called the "binary rev" or "binary reversed".

        Args:
            method (str, optional): the binarization method to apply.
                Defaults to "adaptative".

        Returns:
            NDArray: array where its inner values are 0 or 1
        """
        return 1 - self.binary(method=method)
