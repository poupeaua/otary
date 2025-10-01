"""
Official Citation:
B. Gatos, I. Pratikakis, S.J. Perantonis,
Adaptive degraded document image binarization,
Pattern Recognition,
Volume 39, Issue 3,
2006,
Pages 317-327,
ISSN 0031-3203,
https://doi.org/10.1016/j.patcog.2005.09.010.
(https://www.sciencedirect.com/science/article/pii/S0031320305003821)

From:
https://www.sciencedirect.com/science/article/abs/pii/S0031320305003821
"""

from typing import Optional

import cv2
import numpy as np
from numpy.typing import NDArray

from otary.image.components.transformer.components.binarizer.ops.niblack_like import (
    threshold_niblack_like,
)
from otary.image.utils.background import background_surface_estimation_gatos
from otary.image.utils.local import sum_local, wiener_filter


def gatos_postprocess(thresh: NDArray, lh: float) -> NDArray:
    """Gatos thresholding postprocess function

    Args:
        thresh (NDArray): threshold image
        lh (float): postprocess parameter which should be the length of the
            height of a character in the image

    Returns:
        NDArray: _description_
    """
    n = int(0.15 * lh)
    ksh = 0.9 * n**2
    # ksw = 0.05 * n**2
    ksw1 = 0.35 * n**2
    # dx = 0.25 * n
    # dy = 0.25 * n

    # 6.1. shrink thresholding for each foreground pixel check nb of background pixels
    psh = sum_local(img=thresh, window_size=n)
    shrink_condition = (thresh == 0) & (psh > ksh)
    thresh[shrink_condition] = 1  # set to background

    # 6.2 swell filter - for each background pixel
    # TODO

    # 6.3 swell filter for each background pixel check nb of foreground pixels
    psw1 = sum_local(img=1 - thresh, window_size=n)
    swell2_cond = (thresh == 1) & (psw1 > ksw1)
    thresh[swell2_cond] = 0  # set to foreground

    return thresh


def threshold_gatos(
    img: NDArray,
    q: float = 0.6,
    p1: float = 0.5,
    p2: float = 0.8,
    lh: Optional[float] = None,
    upsampling: bool = False,
    upsampling_factor: int = 2,
    postprocess: bool = False,
) -> NDArray[np.uint8]:
    """Apply Gatos local thresholding.

    Paper (2005):
    https://users.iit.demokritos.gr/~bgat/PatRec2006.pdf

    Args:
        q (float, optional): q gatos factor. Defaults to 0.6.
        p1 (float, optional): p1 gatos factor. Defaults to 0.5.
        p2 (float, optional): p2 gatos factor. Defaults to 0.8.
        lh (Optional[float], optional): height of character.
            Defaults to None, meaning it is computed automatically to be
            a fraction of the image size.
        upsampling (bool, optional): whether to apply gatos upsampling definition.
            Defaults to False.
        upsampling_factor (int, optional): gatos upsampling factor. Defaults to 2.

    Returns:
        NDArray[np.uint8]: thresholded image
    """
    # pylint: disable=too-many-arguments, too-many-positional-arguments
    # pylint: disable=too-many-locals
    if not 0 < q < 1 or not 0 < p1 < 1 or not 0 < p2 < 1:
        raise ValueError("q, p1 and p2 must be in range ]0, 1[")

    if lh is None:  # guess the character height
        im_side = np.sqrt(img.shape[0] * img.shape[1])
        lh = im_side / 20

    # 1. Preprocessing I(x,y) from I_s(x,y) which is input image or source
    im_ = wiener_filter(img=img, window_size=3).astype(np.uint8)

    # 2. Sauvola thresholding S(x,y) with parameters from paper
    s = (
        threshold_niblack_like(img=im_, method="sauvola", window_size=15, k=0.2)[1]
        / 255
    )
    # in paper S(x,y) is in range [0, 1]

    # 3. Background Surface Estimation - B(x,y)
    w_bse = int(2 * lh)
    bg = background_surface_estimation_gatos(img=im_, binary=s, window_size=w_bse)

    # 4. Final Thresholding - T(x,y)
    bg_img_diff = bg - im_
    delta = np.sum(bg_img_diff) / np.sum(1 - s)  # avg distance foreground background
    b = np.sum(bg * s) / np.sum(s)  # avg background value

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def distance_gatos(img: NDArray):
        x = 2 * (2 * img / b - (1 + p1)) / (1 - p1)
        return q * delta * ((1 - p2) * sigmoid(x) + p2)

    if not upsampling:
        thresh = np.where(
            bg_img_diff > distance_gatos(bg),
            0,
            1,
        )
    else:
        # 5. Optional Upsampling using bicubic interpolation
        # using the bicubic interpolation to upsample the base image I(x,y)
        # but using the nearest neighbour to replicate pixels in background B(x,y)
        if upsampling_factor <= 0:
            raise ValueError(
                f"The upsampling factor {upsampling_factor} must be stricly positive"
            )
        if not isinstance(upsampling_factor, int):
            raise ValueError(
                f"The upsampling factor {upsampling_factor} must be an integer"
            )
        i_u = cv2.resize(
            im_,
            None,
            fx=upsampling_factor,
            fy=upsampling_factor,
            interpolation=cv2.INTER_CUBIC,
        )
        b_u = cv2.resize(
            bg,
            None,
            fx=upsampling_factor,
            fy=upsampling_factor,
            interpolation=cv2.INTER_NEAREST,
        )
        thresh = np.where(
            b_u - i_u > distance_gatos(b_u),
            0,
            1,
        )

        # then downsampling to go back to the original size
        thresh = cv2.resize(
            thresh,
            None,
            fx=1 / upsampling_factor,
            fy=1 / upsampling_factor,
            interpolation=cv2.INTER_NEAREST,
        )

    if postprocess:  # 6. post-processing
        thresh = gatos_postprocess(thresh=thresh, lh=lh)

    img_thresholded = thresh.astype(np.uint8) * 255
    return img_thresholded
