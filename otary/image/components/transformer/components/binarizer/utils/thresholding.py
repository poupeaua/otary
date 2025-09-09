"""
Thresholding techniques
"""

from typing import Optional
import cv2
import numpy as np
from numpy.typing import NDArray

from otary.image.utils.local import (
    high_contrast_local,
    max_local,
    mean_local,
    min_local,
    sum_local,
    wiener_filter,
    windowed_mult,
)
from otary.image.utils.tools import bwareaopen, check_transform_window_size


def threshold_niblack_like(
    img: np.ndarray,
    method: str = "sauvola",
    window_size: int = 15,
    k: float = 0.5,
    r: float = 128.0,
) -> tuple[NDArray, NDArray[np.uint8]]:
    """Fast implementation of the Niblack-like thresholdings.
    These thresholdings are similar so we just put them in the same utils function.

    These thresholding methods are local thresholding methods. This means that
    the threshold value is computed for each pixel based on the pixel values
    in a window around the pixel. The window size is a parameter of the function.

    Local thresholding methods are generally better than global thresholding
    methods (like Otsu or adaptive thresholding) for images with varying
    illumination.

    It includes the following methods:
    - Niblack
    - Sauvola
    - Wolf
    - Nick
    - WAN
    - Singh

    Originally, the sauvola thresholding was invented for text recognition like
    most of the niblack-like thresholding methods.

    Args:
        img (np.ndarray): image inputs
        method (str, optional): method to apply.
            Must be in ["niblack", "sauvola", "nick", "wolf"]. Defaults to "sauvola".
        window_size (int, optional): window size. Defaults to 15.
        k (float, optional): k factor. Defaults to 0.5.
        r (float, optional): r value used only in sauvola. Defaults to 128.0.

    Returns:
        tuple[NDArray, NDArray[np.uint8]]: thresh and thresholded image
    """
    window_size = check_transform_window_size(img, window_size)

    img = img.astype(np.float32)

    # compute intensity representation of image
    mean = mean_local(img=img, window_size=window_size)
    sqmean = mean_local(img=img**2, window_size=window_size)
    var = sqmean - mean**2
    std = np.sqrt(np.clip(var, 0, None))

    if method == "niblack":
        thresh = mean + k * std
    elif method == "sauvola":
        thresh = mean * (1 + k * ((std / r) - 1))
    elif method == "wan":
        wan_mean = (max_local(img=img, window_size=window_size) + mean) / 2
        thresh = wan_mean * (1 + k * ((std / r) - 1))
    elif method == "wolf":
        max_std = np.max(
            [std, np.full_like(std, 1e-5)]
        )  # local & 1e-5 to avoid division by zero
        min_img = np.min(img)  # global
        thresh = mean - k * (1 - (std / max_std)) * (mean - min_img)
    elif method == "nick":
        thresh = mean + k * np.sqrt(var + mean**2)
    elif method == "singh":
        # essentially this is Sauvola with an approximation to compute the
        # local standard deviation to improve speed
        std_local_approx = img - mean
        thresh = mean * (1 + k * (std_local_approx / (1 - std_local_approx + 1e-9) - 1))
    else:
        raise ValueError(f"Unknown method {method} for threshold_niblack_like")

    # compute the output, meaning the threshold and the thresholded image
    img_thresholded = (img > thresh).astype(np.uint8) * 255

    return thresh, img_thresholded


def threshold_bernsen(
    img: NDArray,
    window_size: int = 75,
    contrast_limit: float = 25,
    threshold_global: int = 100,
) -> NDArray[np.uint8]:
    """Implementation of the Bernsen thresholding method.

    This is a local thresholding method that computes the threshold for a pixel
    based on a small region around it.

    Args:
        img (NDArray): input image
        window_size (int, optional): window size for local computations.
            Defaults to 75.
        contrast_limit (float, optional): contrast limit. If the
            contrast is higher than this value, the pixel is thresholded by the
            bernsen threshold otherwise the global threshold is used.
            Defaults to 25.
        threshold_global (int, optional): global threshold. Defaults to 100.

    Returns:
        NDArray[np.uint8]: output thresholded image
    """
    z_high = max_local(img=img, window_size=window_size)
    z_low = min_local(img=img, window_size=window_size)
    bernsen_contrast = z_high - z_low
    bernsen_threshold = (z_high + z_low) / 2

    threshold_local = np.where(
        bernsen_contrast > contrast_limit,
        bernsen_threshold,
        threshold_global,  # global threshold is broadcast
    )

    img_thresholded = (img > threshold_local).astype(np.uint8) * 255

    return img_thresholded


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
    window_size = check_transform_window_size(img, window_size)
    contrast_window_size = check_transform_window_size(img, contrast_window_size)

    # step 1: Initialization step -> High Contrast Image Construction
    I_c = high_contrast_local(img=img, window_size=contrast_window_size)

    # step 1.b: Opening operation
    # is optional because it generally removes too much details
    if opening_n_min_pixels > 0:
        I_c = bwareaopen(
            I_c, n_min_pixels=opening_n_min_pixels, connectivity=opening_connectivity
        )

    # step 2: Sauvola’s Binarization Step
    _, I_s = threshold_niblack_like(
        img=img, method="sauvola", window_size=window_size, k=k, r=r
    )

    # reverse so that I_c and I_s both fit in terms of values 0 and 255
    I_s = 255 - I_s

    # step 3: Sequential Combination
    # for all pixels p in I_c:
    # -- if I_c(p) == true:
    # ---- detect the set of pixels overlapping with p in I_s
    # the pixels overlapping (pixels in plural!) are connected components
    _, cc_labels_matrix = cv2.connectedComponents(I_s, connectivity=connectivity)
    overlapping_pixels = (I_c == 255) * cc_labels_matrix
    cc_labels_with_overlap = list(set(np.unique(overlapping_pixels)) - {0})
    mask = np.isin(element=cc_labels_matrix, test_elements=cc_labels_with_overlap)

    return 255 - (I_s * mask)  # reverse once again because we already reversed I_s


def threshold_su(
    img: NDArray,
    window_size: int = 3,
    n_min: int = -1,
) -> NDArray[np.uint8]:
    """Compute the Su local thresholding.

    Paper (2010):
    https://www.researchgate.net/publication/220933012

    Args:
        img (NDArray): input grayscale image
        window_size (int, optional): window size for local computation. Defaults to 3.
        n_min (int, optional): minimum number of high contrast pixels within the
            neighborhood window. Defaults to -1 meaning that n_min = window_size.

    Returns:
        NDArray[np.uint8]: output thresholded image
    """
    if n_min < 0:
        n_min = window_size

    I_c = high_contrast_local(img=img, window_size=window_size)

    # number of high contrast pixels
    N_e = sum_local(img=I_c.astype(np.float32) / 255, window_size=window_size) + 1e-9

    tmp = (I_c == 255) * img
    img_sum = sum_local(img=tmp, window_size=window_size)
    E_mean = img_sum / N_e

    E_std = np.sqrt((img_sum - N_e * E_mean) ** 2 / 2)

    cond = (N_e >= n_min) & (img <= E_mean + E_std / 2)
    return np.where(cond, 0, 255).astype(np.uint8)


def threshold_gatos(
    img: NDArray,
    q: float = 0.6,
    p1: float = 0.5,
    p2: float = 0.8,
    lh: Optional[float] = None,
    upsampling: bool = False,
    upsampling_factor: int = 2
) -> NDArray[np.uint8]:
    """

    Important note: the S(x,y) in paper is in range [0, 1] and 1 corresponds to 
    foreground (or close to 0 values) whereas 0 corresponds to background.
    In this implementation we use the opposite values (0 -> foreground, 1 -> background)
    So when in paper use (1-S) here use S.

    Args:
        img (NDArray): _description_

    Returns:
        NDArray[np.uint8]: _description_
    """
    if not (0 < q < 1) or not (0 < p1 < 1) or not (0 < p2 < 1):
        raise ValueError("q, p1 and p2 must be in range ]0, 1[")
    
    if lh is None: # guess the character height
        im_side = np.sqrt(img.shape[0] * img.shape[1])
        lh = im_side / 20

    # 1. Preprocessing I(x,y) from I_s(x,y) which is input image or source
    I = wiener_filter(img=img, window_size=3)

    # 2. Sauvola thresholding S(x,y) with parameters from paper
    S = threshold_niblack_like(img=I, method="sauvola", window_size=15, k=0.2)
    S = S / 255 # important since in paper S(x,y) is in range [0, 1]

    # 3. Background Surface Estimation - B(x,y)
    bse = windowed_mult(img=I, img2=S, window_size=3) / (S + 1e-9)

    B = np.where(
        S == 0,
        bse,
        I,
    )

    # 4. Final Thresholding - T(x,y)
    bg_img_diff = B - I
    delta = np.sum(bg_img_diff) / np.sum(1 - S) # avg distance foreground background
    b = np.sum(B * S) / np.sum(S) # avg background value

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def distance_gatos(img: NDArray):
        x = 2 * (2 * img / b - (1 + p1)) / (1 - p1)
        return q * delta * ((1 - p2) * sigmoid(x) + p2)

    if not upsampling:
        T = np.where(
            bg_img_diff > distance_gatos(B),
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
        M = upsampling_factor
        I_u = cv2.resize(I, None, fx=M, fy=M, interpolation=cv2.INTER_CUBIC)
        B_u = cv2.resize(B, None, fx=M, fy=M, interpolation=cv2.INTER_NEAREST)
        T = np.where(
            B_u - I_u > distance_gatos(B_u),
            0,
            1,
        )

    # 6. post-processing
    lh = 2
    n = 0.15 * lh
    ksh = 0.9 * n**2
    ksw = 0.05 * n**2
    ksw1 = 0.35 * n**2
    dx = 0.25 * n
    dy = 0.25 * n

    # 6.1. shrink thresholding - for each foreground pixel
    Psh = sum_local(img=1-T, window_size=n)
    shrink_condition = (T == 0) & (Psh > ksh)
    T[shrink_condition] = 1  # set to background

    # 6.2 swell filter - for each background pixel
    h, w = T.shape
    Y, X = np.mgrid[0:h, 0:w]  # coordinate grids

    # Count foreground pixels in each window
    Psw = sum_local(img=T, window_size=n)

    # Sum of x and y coordinates of foreground pixels in window
    sumX = sum_local(img=X * T, window_size=n)
    sumY = sum_local(img=Y * T, window_size=n)

    # Avoid division by zero
    xa = np.zeros_like(sumX, dtype=np.float32)
    ya = np.zeros_like(sumY, dtype=np.float32)
    mask_valid = Psw > 0
    xa[mask_valid] = sumX[mask_valid] / Psw[mask_valid]
    ya[mask_valid] = sumY[mask_valid] / Psw[mask_valid]

    # Conditions for swelling
    cond = (
        (T == 0) &  # only background pixels
        (Psw > ksw) &
        (np.abs(X - xa) < dx) &
        (np.abs(Y - ya) < dy)
    )

    # Apply
    T[cond] = 1

    # 6.3 swell filter once again - for each background pixel
    Psw1 = sum_local(img=T, window_size=n)
    shrink_condition = (T == 0) & (Psw1 > ksw1)
    T[shrink_condition] = 1  # set to background

    return T


