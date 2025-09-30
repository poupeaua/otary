"""
Pre-processing step in the FAIR algorithm.
"""

import cv2
import numpy as np
from numpy.typing import NDArray

from otary.image.components.transformer.components.binarizer.ops.fair.constant import (
    UNKNOWN_LABEL,
)
from otary.image.utils.local import gradient_magnitude
from otary.image.components.transformer.components.binarizer.ops.fair.clustering.fair_cluster import (
    fair_clustering,
)


def threshold_sfair(
    img: NDArray,
    k: float = 1.0,
    alpha: float = 0.5,
    n: int = 15,
    clustering_algo: str = "kmeans",
    clustering_max_iter: int = 10,
    thining: float = 1.0,
) -> NDArray:
    """S-FAIR thresholding method.
    This typically serves as a building block for the FAIR method.

    However it can be also used as a standalone binarization method as it is still
    useful as-is and is typically much faster than running the whole FAIR algorithm.

    Args:
        img (NDArray): input image
        k (float, optional): _description_. Defaults to 1.0.
        alpha (float, optional): It defines the ratio to compute the lower threshold
            in the 1st step of the S-FAIR step. It is generally in [0.3, 0.5].
            Defaults to 0.4.
        n (int, optional): window size for the EM algorithm and hence to cluster
            background and foreground pixels around edge pixels.
            This parameter is important as a higher value will make the method
            more robust to noise but also more computationally expensive and slow.
            Defaults to 51.
        clustering_algo (str, optional): clustering algorithm to use.
            Clustering algorithms implemented: "kmeans", "em" and "otsu".
            em stands for Expectation Maximization.
            Defaults to "kmeans".
        clustering_max_iter (int, optional): maximum number of iterations for the
            clustering algorithm. Defaults to 10.
        thining (float, optional): thining factor in [0, 1]. 0 means no thinning which
            means that all edge pixels are processed. 1 means that only every
            n // 2 edge pixels are processed which signicantly speeds up the
            algorithm but can make it a bit less accurate. Defaults to 1.0.

    Returns:
        NDArray: A 3-values-binarized image with 0, 0.5 and 1 values.
            binarized image with three values not 2 as it is most common for
            binarization methods. Here 0 means background, 0.5 means unknown and
            1 means foreground.
    """
    # Step 1 of S-FAIR - Text area detection
    gm = gradient_magnitude(img=img, window_size=3)
    T_o = cv2.threshold(gm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    T_u = k * T_o  # T_u stands for upper threshold
    T_l = alpha * T_o  # T_l stands for lower threshold
    img_edged = cv2.Canny(image=img, threshold1=T_l, threshold2=T_u)  # values 0 or 255
    s = np.column_stack(np.where(img_edged == 255))  # edge pixel coordinates

    # to speed up EM, we can downsample the number of edge pixels to process
    # thining must be in [n // 2, 1]
    thining = n // int(n * (1 - thining) + 2 * thining)
    s = s[::thining]  # downsample edge pixels to speed up EM

    # Step 2 of S-FAIR - Model estimation around edges
    # edges can be easily identified as they are 255 pixels in im_edges
    # get patches for vectorized computation - trick do padding to get all patches
    # of same size even when the center of edge pixel is at border
    pad = n // 2
    img_pad = cv2.copyMakeBorder(img, pad, pad, pad, pad, borderType=cv2.BORDER_DEFAULT)
    patches = np.lib.stride_tricks.sliding_window_view(x=img_pad, window_shape=(n, n))
    x = patches[s[:, 0], s[:, 1]]  # shape (n_edges, n, n)
    gamma = fair_clustering(
        x=x, max_iter=clustering_max_iter, algorithm=clustering_algo
    )

    # compute the mean responsability for each pixel
    # to get a smoother result (as each pixel can be in multiple patches)
    resp_sum = np.zeros_like(img_pad, dtype=np.float32)
    resp_count = np.zeros_like(img_pad, dtype=np.float32)
    for i, (r, c) in enumerate(s):
        resp_sum[r : r + n, c : c + n] += gamma[i]
        resp_count[r : r + n, c : c + n] += 1
    gamma_tilde = np.divide(resp_sum, resp_count, where=resp_count != 0)

    # compute z_i = 1 if gamma_tilde > 0.5 else 0
    z = np.where(gamma_tilde > 0.5, 1, 0).astype(np.float32)
    z[resp_count == 0] = UNKNOWN_LABEL
    z = z[pad:-pad, pad:-pad]  # remove padding to get back to original img size

    return z
