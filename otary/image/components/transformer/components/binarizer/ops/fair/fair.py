"""
Official Citation:
Thibault Lelore, Frédéric Bouchara. FAIR: A Fast Algorithm for document Image
Restoration.
IEEE Transactions on Pattern Analysis and Machine Intelligence, 2013, 35 (8),
pp.2039-2048.
ff10.1109/TPAMI.2013.63ff. ffhal-01479805f

From:
https://amu.hal.science/hal-01479805/document
"""

import numpy as np
from numpy.typing import NDArray

from otary.image.components.transformer.components.binarizer.ops.fair.postprocessing import (
    correct_misclassified_text_pixels,
    final_labeling,
    remove_stains,
)
from otary.image.components.transformer.components.binarizer.ops.fair.sfair import (
    threshold_sfair,
)


def threshold_fair(
    img: NDArray,
    sfair_window_size: int = 33,
    sfair_clustering_algo: str = "otsu",
    sfair_clustering_max_iter: int = 50,
    sfair_thining: float = 1.0,
    sfair_alpha: float = 0.38,
    post_stain_max_pixels: int = 25,
    post_misclass_txt: bool = True,
    post_clustering_algo: str = "otsu",
    post_clustering_max_iter: int = 10,
    post_max_iter: int = 15,
    post_window_size: int = 75,
    post_beta: float = 1.0,
) -> NDArray:
    """FAIR thresholding method.

    Implementation comes from the following paper:
    https://amu.hal.science/hal-01479805/document

    In the official paper they propose the use of two clustering algorithms:
    - Expectation Maximization (EM) algorithm: to cluster background and
    foreground pixels around edge pixels by assuming that the distribution is
    a mixture of two gaussians. The computation is done in the 1D case thus only
    considering a input grayscale image as parameter img.
    - K-Means algorithm: to cluster background and foreground pixels around edge
    with no assumption on any distribution.

    Here we add the option to use Otsu thresholding as a clustering algorithm.
    With this additional option, the algorithm is much faster and provide excellent
    results. It is like combining the idea of the AdOtsu algorithm (that runs
    Otsu on smaller patches or grids that divide the image) and the FAIR idea of running
    the clustering only on a window around the edge pixels.

    Expectation-Maximization is an excellent option to run this algorithm however
    the choice of the sfair window size is critical. A small sfair window size
    should be consider (5-10) to get good results. A big sfair window size (> 75)
    will make the algorithm run very slow but can be useful too depending of the
    type of document.

    K-Means is an interesting option to run this algorithm however the choice of the
    postprocess window size is critical too and seems to provide better results with
    a big sfair window size (> 100)

    Otsu seems like a robust option. It does not seem to be too sensitive to the
    choice of the sfair window size.

    Two main configurations are interesting with this method:

    >>> # For whole page document
    >>> bin = threshold_fair(
    >>>     img=im.asarray,
    >>>     sfair_window_size=5,
    >>>     sfair_em_max_iter=100,
    >>>     sfair_em_thining_factor=1.0,
    >>>     postprocess_em_max_iter=15,
    >>>     postprocess_max_iter=10,
    >>>     postprocess_window_size=75,
    >>> )

    >>> # High S-FAIR window size -> can detect very thin text components
    >>> # is good for images with text already zoomed in (only a sentence for example)
    >>> bin = threshold_fair(
    >>>     img=im.asarray,
    >>>     sfair_window_size=75,
    >>>     sfair_em_max_iter=15,
    >>>     sfair_em_thining_factor=1.0,
    >>>     postprocess_em_max_iter=15,
    >>>     postprocess_max_iter=10,
    >>>     postprocess_window_size=35,
    >>> )

    >>> # Low S-FAIR window size -> insist on more local text contours
    >>> bin = threshold_fair(
    >>>     img=im.asarray,
    >>>     sfair_window_size=15,
    >>>     sfair_em_max_iter=25,
    >>>     sfair_em_thining_factor=1.0,
    >>>     postprocess_em_max_iter=15,
    >>>     postprocess_max_iter=10,
    >>>     postprocess_window_size=35,
    >>> )

    Args:
        img (NDArray): input grayscale image of shape (H, W)
        sfair_window_size (int, optional): window size in preprocess
            to cluster background and foreground pixels around edge pixels.
            This parameter is important as a higher value will make the method
            more robust to noise but also more computationally expensive and slow.
            Defaults to 5.
        sfair_clustering_algo (str, optional): clustering algorithm for the S-FAIR
            step. Defaults to "kmeans".
        sfair_clustering_max_iter (int, optional): maximum number of iterations for the
            clustering algorithm within the S-FAIR step. Defaults to 50.
        sfair_thining (float, optional): thining factor in [0, 1]. 0 means no thinning
            which means that all edge pixels are processed. 1 means that only every
            sfair_window_size // 2 edge pixels are processed which signicantly speeds
            up the computation. Defaults to 1.0.
        sfair_alpha (float, optional): It defines the ratio to compute the lower
            threshold in the 1st step of the S-FAIR step. It is generally in [0.3, 0.5].
            Defaults to 0.38.
        post_stain_max_pixels (int, optional): maximum number of pixels for a stain to
            be considered as an unknown connected component. Defaults to 25.
        post_misclass_txt (bool, optional): whether to perform the
            post-processing correct_misclassified_text_pixels step. Defaults to True.
        post_clustering_algo (str, optional): clustering algorithm for the
            post-processing step. Defaults to "kmeans".
        post_clustering_max_iter (int, optional): maximum number of iterations for the
            EM algorithm within the post-processing step. Defaults to 10.
        post_max_iter (int, optional): maximum number of iterations for the
                correct_misclassified_text_pixels step within the post-processing step.
                Defaults to 15.
        post_window_size (int, optional): window size in postprocess
            to cluster background and foreground pixels around edge pixels.
            This parameter is important as a higher value will make the method
            more robust to noise but also more computationally expensive and slow.
            Defaults to 75.
        post_beta (float, optional): factor to define if the unkown pixels
            should be set as text or background. If beta is 1 then
            unknown pixels are set to text if the number of surrounding text pixels
            (N_t) is higher than the number of surrounding background pixels (N_b).
            Simply N_t > N_b. Beta is the value to put more flexibility on the rule
            and thus set unknown pixels to text if N_t > beta * N_b
            Defaults to 1.0.

    Returns:
        NDArray: binarized image
    """
    # Step 1 - Double Thresholding: compute S-FAIR twice with different k
    z1 = threshold_sfair(
        img=img,
        k=1.4,
        alpha=sfair_alpha,
        n=sfair_window_size,
        clustering_algo=sfair_clustering_algo,
        clustering_max_iter=sfair_clustering_max_iter,
        thining=sfair_thining,
    )
    z2 = threshold_sfair(
        img=img,
        k=1.66,
        alpha=sfair_alpha,
        n=sfair_window_size,
        clustering_algo=sfair_clustering_algo,
        clustering_max_iter=sfair_clustering_max_iter,
        thining=sfair_thining,
    )

    # Step 2 - Merging: combine the two S-FAIR results
    I_m = np.max([z1, z2], axis=0)  # values in {0, 0.5, 1}

    # ------------------------------ POSTPROCESSING -----------------------------------

    # Step 3 - Post-Filtering process
    # Step 3.a. - remove stains connected components only surrounded by unknown pixels
    I_m = remove_stains(arr=I_m, stain_max_pixels=post_stain_max_pixels)

    # Step 3.b. - correct misclassified text pixels
    if post_misclass_txt:
        I_m = correct_misclassified_text_pixels(
            I_m=I_m,
            img=img,
            n=post_window_size,
            clustering_algo=post_clustering_algo,
            clustering_max_iter=post_clustering_max_iter,
            max_iter=post_max_iter,
        )

    # Step 4. - Final labeling
    I_m = final_labeling(I_m=I_m, beta=post_beta)

    I_m = (1 - I_m).astype(np.uint8) * 255  # reverse now 0 is text and 1 is background

    return I_m
