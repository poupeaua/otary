"""
Official Citation:
Hadjadj, Zineb & Meziane, Abdelkrim & Cherfa, Yazid & Cheriet, Mohamed & Setitra,
Insaf. (2016).
ISauvola: Improved Sauvola’s Algorithm for Document Image Binarization.
9730. 737-745. 10.1007/978-3-319-41501-7_82.

From:
https://www.researchgate.net/publication/304621554
"""

import cv2
import numpy as np
from numpy.typing import NDArray

from otary.image.components.transformer.components.binarizer.ops.niblack_like import (
    threshold_niblack_like,
)
from otary.image.utils.local import high_contrast_local
from otary.image.utils.tools import bwareaopen, check_transform_window_size


def threshold_isauvola(
    img: NDArray,
    window_size: int = 15,
    k: float = 0.01,
    r: float = 128.0,
    connectivity: int = 8,
    contrast_window_size: int = 3,
    opening_n_min_pixels: int = 0,
    opening_connectivity: int = 8,
) -> NDArray[np.uint8]:
    """Implementation of the ISauvola thresholding method.

    This is a local thresholding method that computes the threshold for a pixel
    based on a small region around it.

    Comes from the article:
    https://www.researchgate.net/publication/304621554

    Args:
        img (NDArray): input image
        window_size (int, optional): Sauvola window size. Defaults to 15.
        k (float, optional): Sauvola k factor. Defaults to 0.5.
        r (float, optional): Sauvola r value. Defaults to 128.0.
        connectivity (int, optional): ISauvola connectivity. Defaults to 8.
        contrast_window_size (int, optional): ISauvola contrast window size.
            Defaults to 3.
        opening_n_min_pixels (float, optional): ISauvola opening n min pixels.
            Defaults to 0.
        opening_connectivity (int, optional): ISauvola opening connectivity.
            Defaults to 8.

    Returns:
        NDArray[np.uint8]: ISauvola thresholded image
    """
    # pylint: disable=too-many-arguments, too-many-positional-arguments
    # pylint: disable=too-many-locals
    window_size = check_transform_window_size(img, window_size)
    contrast_window_size = check_transform_window_size(img, contrast_window_size)

    # step 1: Initialization step -> High Contrast Image Construction
    i_c = high_contrast_local(img=img, window_size=contrast_window_size)

    # step 1.b: Opening operation
    # is optional because it generally removes too much details
    if opening_n_min_pixels > 0:
        i_c = bwareaopen(
            i_c, n_min_pixels=opening_n_min_pixels, connectivity=opening_connectivity
        )

    # step 2: Sauvola’s Binarization Step
    _, i_s = threshold_niblack_like(
        img=img, method="sauvola", window_size=window_size, k=k, r=r
    )

    # reverse so that I_c and I_s both fit in terms of values 0 and 255
    i_s = 255 - i_s

    # step 3: Sequential Combination
    # for all pixels p in I_c:
    # -- if I_c(p) == true:
    # ---- detect the set of pixels overlapping with p in I_s
    # the pixels overlapping (pixels in plural!) are connected components
    _, cc_labels_matrix = cv2.connectedComponents(i_s, connectivity=connectivity)
    overlapping_pixels = (i_c == 255) * cc_labels_matrix
    cc_labels_with_overlap = list(set(np.unique(overlapping_pixels)) - {0})
    mask = np.isin(element=cc_labels_matrix, test_elements=cc_labels_with_overlap)

    return 255 - (i_s * mask)  # reverse once again because we already reversed I_s
