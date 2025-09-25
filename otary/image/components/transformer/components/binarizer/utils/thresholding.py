"""
Thresholding techniques
"""

from typing import Optional

import cv2
import numpy as np
from numpy.typing import NDArray

from otary.image.utils.background import (
    background_surface_estimation_adotsu,
    background_surface_estimation_gatos,
)
from otary.image.utils.grid import grid_view, ungrid
from otary.image.utils.local import (
    high_contrast_local,
    max_local,
    mean_local,
    min_local,
    sum_local,
    variance_local,
    wiener_filter,
)
from otary.image.utils.tools import bwareaopen, check_transform_window_size


def threshold_niblack_like(
    img: np.ndarray,
    method: str = "sauvola",
    window_size: int = 15,
    k: float = 0.5,
    r: float = 128.0,
    p: float = 3.0,
    q: float = 10.0,
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
    - Phansalkar
    - WAN
    - Wolf
    - Nick
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
        p (float, optional): p value used only in Phansalkar et al. method.
            Defaults to 3.0.
        q (float, optional): q value used only in Phansalkar et al. method.
            Defaults to 10.0

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
    elif method == "phansalkar":
        thresh = mean * (1 + p * np.exp(-q * mean) + k * ((std / r) - 1))
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
        thresh = mean + k * np.sqrt(sqmean)  # sqmean = var + mean**2 = B + m^2
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


def threshold_feng(
    img: NDArray,
    w1: int = 19,
    w2: int = 33,
    alpha1: float = 0.12,
    k1: float = 0.25,
    k2: float = 0.04,
    gamma: float = 2.0,
) -> NDArray[np.uint8]:
    """Implementation of the Feng thresholding method.

    Paper (2004):
    https://www.jstage.jst.go.jp/article/elex/1/16/1_16_501/_pdf

    Args:
        img (NDArray): input grayscale image
        w1 (int, optional): primary window size. Defaults to 19.
        w2 (int, optional): secondary window value. Defaults to 33.
        alpha1 (float, optional): alpha1 value. Defaults to 0.12.
        k1 (float, optional): k1 value. Defaults to 0.25.
        k2 (float, optional): k2 value. Defaults to 0.04.
        gamma (float, optional): gamma value. Defaults to 2.0.

    Returns:
        NDArray[np.uint8]: output thresholded image
    """
    # pylint: disable=too-many-arguments, too-many-positional-arguments
    if not (0 < w1 < w2):
        raise ValueError("Using Feng thresholding requires 0 < w1 < w2")

    # mean local on primary window
    m = mean_local(img=img, window_size=w1)

    # min local on primary window
    M = min_local(img=img, window_size=w1)

    # std local on primary window
    sqmean = mean_local(img=img**2, window_size=w1)
    var = sqmean - m**2
    s = np.sqrt(np.clip(var, 0, None))

    # std in local secondary window
    Rs = variance_local(img=img, window_size=w2)

    # setup parameters
    normalized_std = s / (Rs + 1e-9)
    alpha2 = k1 * (normalized_std**gamma)
    alpha3 = k2 * (normalized_std**gamma)

    # compute threshold
    T = (1 - alpha1) * m + alpha2 * normalized_std * (m - M) + alpha3 * M
    T = np.clip(T, 0, 255)

    img_thresholded = (img > T).astype(np.uint8) * 255
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


def gatos_postprocess(T: NDArray, lh: float) -> NDArray:
    n = int(0.15 * lh)
    ksh = 0.9 * n**2
    # ksw = 0.05 * n**2
    ksw1 = 0.35 * n**2
    # dx = 0.25 * n
    # dy = 0.25 * n

    # 6.1. shrink thresholding for each foreground pixel check nb of background pixels
    Psh = sum_local(img=T, window_size=n)
    shrink_condition = (T == 0) & (Psh > ksh)
    T[shrink_condition] = 1  # set to background

    # 6.2 swell filter - for each background pixel
    # TODO

    # 6.3 swell filter for each background pixel check nb of foreground pixels
    Psw1 = sum_local(img=1 - T, window_size=n)
    swell2_cond = (T == 1) & (Psw1 > ksw1)
    T[swell2_cond] = 0  # set to foreground

    return T


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
    if not (0 < q < 1) or not (0 < p1 < 1) or not (0 < p2 < 1):
        raise ValueError("q, p1 and p2 must be in range ]0, 1[")

    if lh is None:  # guess the character height
        im_side = np.sqrt(img.shape[0] * img.shape[1])
        lh = im_side / 20

    # 1. Preprocessing I(x,y) from I_s(x,y) which is input image or source
    I_ = wiener_filter(img=img, window_size=3)

    # 2. Sauvola thresholding S(x,y) with parameters from paper
    S = threshold_niblack_like(img=I_, method="sauvola", window_size=15, k=0.2)[1] / 255
    # in paper S(x,y) is in range [0, 1]

    # 3. Background Surface Estimation - B(x,y)
    w_bse = int(2 * lh)
    B = background_surface_estimation_gatos(img=I_, binary=S, window_size=w_bse)

    # 4. Final Thresholding - T(x,y)
    bg_img_diff = B - I_
    delta = np.sum(bg_img_diff) / np.sum(1 - S)  # avg distance foreground background
    b = np.sum(B * S) / np.sum(S)  # avg background value

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
        I_u = cv2.resize(I_, None, fx=M, fy=M, interpolation=cv2.INTER_CUBIC)
        B_u = cv2.resize(B, None, fx=M, fy=M, interpolation=cv2.INTER_NEAREST)
        T = np.where(
            B_u - I_u > distance_gatos(B_u),
            0,
            1,
        )

        # then downsampling to go back to the original size
        T = cv2.resize(T, None, fx=1 / M, fy=1 / M, interpolation=cv2.INTER_NEAREST)

    if postprocess:  # 6. post-processing
        T = gatos_postprocess(T=T, lh=lh)

    img_thresholded = T.astype(np.uint8) * 255
    return img_thresholded


def threshold_bradley_roth(
    img: NDArray, window_size: int = 15, t: float = 0.15
) -> NDArray[np.uint8]:
    """Implementation of the Bradley & Roth thresholding method.
    This is actually a very easy thresholding method solely depending on the
    local mean.

    This is a local thresholding method that computes the threshold for a pixel
    based on a small region around it.

    Args:
        img (NDArray): input image
        window_size (int, optional): window size for local computations.
            Defaults to 15.
        t (float, optional): t value in [0, 1]. Defaults to 0.15.

    Returns:
        NDArray[np.uint8]: output thresholded image
    """
    if not (0 < t < 1):
        raise ValueError("t must be in range ]0, 1[")

    m = mean_local(img=img, window_size=window_size)
    img_thresholded = (img > m * (1 - t)).astype(np.uint8) * 255
    return img_thresholded


def threshold_adotsu(
    img: NDArray,
    grid_size: int = 50,
    k_sigma: float = 1.6,
    n_steps: int = 2,
    is_multiscale_enabled: bool = False,
    eps: float = 1e-9,
) -> NDArray[np.uint8]:
    r"""Implementation of the AdOtsu thresholding method.

    AdOtsu is computed this way on a grid-based approach:

    $$
    T_{AdOtsu, u}(x) = \epsilon + (Otsu(x) + \epsilon) \times \Theta(\sigma(x) - k_{\sigma}\sigma_{EB}(x))
    $$

    Currently the Multi-Scale AdOtsu is not implemented.
    It only computes using the basic AdOtsu pipeline.
    Still the Background Surface Estimation is computed using a multiscale approach
    as described in the paper.

    Args:
        img (NDArray): input image
        grid_size (int, optional): window size for local computations.
            Defaults to 15.
        k_sigma (float, optional): k_sigma value in [1, 2]. Defaults to 1.6.
        is_multiscale_enabled (bool, optional): is multiscale enabled.
            Defaults to False.
        eps (float, optional): epsilon value to avoid division by zero.
            Defaults to 1e-9.

    Returns:
        NDArray[np.uint8]: output thresholded image
    """
    if is_multiscale_enabled:
        raise NotImplementedError("Multiscale is not implemented for AdOtsu")

    # produce the rough binarization
    binary = threshold_niblack_like(img=img, method="sauvola", k=0.1)[1]
    binary_bool = cv2.erode(binary, np.ones((3, 3)), iterations=4) / 255

    for i in range(n_steps):  # update the binarized image to a more accurate one
        binary = adotsu_single_step(
            img=img, binary=binary_bool, grid_size=grid_size, k_sigma=k_sigma, eps=eps
        )

        if i != n_steps - 1:  # while not last step turn the binary into bool
            binary_bool = binary / 255

    return binary


def adotsu_single_step(
    img: NDArray, binary: NDArray, grid_size: int, k_sigma: float, eps: float
) -> NDArray[np.uint8]:
    bse = background_surface_estimation_adotsu(
        img=img, binary=binary, grid_size_init=grid_size
    )

    # grid based view as patches
    pad = -1
    img_grid = grid_view(arr=img, grid_size=grid_size, pad_value=pad)
    bse_grid = grid_view(arr=bse, grid_size=grid_size, pad_value=pad)

    # otsu threshold in each patch
    otsu = otsu_grid_based(patches=img_grid, excluding_padding_values=pad)

    # variance calculation for each patch in both img and bse
    var_img = np.var(img_grid, axis=(2, 3), where=img_grid != pad)
    var_bse = np.var(bse_grid, axis=(2, 3), where=bse_grid != pad)
    step_fn = np.where(var_img > k_sigma * var_bse, 1, 0)

    # AdOtsu threshold
    adotsu_t = eps + (otsu + eps) * step_fn

    # bring back to the original image shape
    tmp = np.repeat(
        np.repeat(adotsu_t[:, :, None, None], grid_size, axis=2), grid_size, axis=3
    )
    T = ungrid(
        blocks=tmp,
        grid_size=grid_size,
        original_shape=(img.shape[0], img.shape[1]),
    ).astype(np.uint8)

    # intermediate step for threshold value interpolation after grid-based approach
    grid_size_odd = grid_size + 1 if grid_size % 2 == 0 else grid_size
    T = cv2.GaussianBlur(T, (grid_size_odd, grid_size_odd), 0)

    img_thresholded = (img > T).astype(np.uint8) * 255
    return img_thresholded


def otsu_grid_based(
    patches: NDArray, nbins: int = 256, excluding_padding_values: int = -1
):
    """
    Compute Otsu threshold for each patch in (num_blocks_h, num_blocks_w, grid, grid).
    Assumes patches are uint8 images.
    """
    nbh, nbw, gh, gw = patches.shape
    nblocks = nbh * nbw
    flat = patches.reshape(nblocks, gh * gw)

    # Exclude values -1 from histogram computation
    hist = np.zeros((nblocks, nbins), dtype=np.int32)
    for i in range(nblocks):
        vals = flat[i]
        vals = vals[vals != excluding_padding_values]
        if vals.size > 0:
            hist[i] = np.bincount(vals, minlength=nbins)
        else:
            hist[i] = 0  # If all values are excluding_padding_values, histogram is zero

    # normalize to probabilities
    hist = hist.astype(np.float32)
    hist_sum = hist.sum(axis=1, keepdims=True)
    # Avoid division by zero
    hist_sum[hist_sum == 0] = 1
    hist = hist / hist_sum

    # cumulative sums (class probabilities)
    omega = np.cumsum(hist, axis=1)
    mu = np.cumsum(hist * np.arange(nbins), axis=1)

    mu_t = mu[:, -1]  # total mean

    # between-class variance for all thresholds
    sigma_b = (mu_t[:, None] * omega - mu) ** 2 / (omega * (1 - omega) + 1e-10)

    # best threshold = argmax variance
    thresholds = sigma_b.argmax(axis=1).astype(np.uint8)

    return thresholds.reshape(nbh, nbw)
