"""
---------------------------------------------------------------------------------------
Official citation for Niblack binarization:
P, A. (1985). An introduction to digital image processing. Birkeroed: Strandberg, 1985.

From:
https://www.academia.edu/1088460/An_introduction_to_digital_image_processing
---------------------------------------------------------------------------------------
Official citation for Sauvola binarization:
Sauvola, Jaakko & Seppänen, Tapio & Haapakoski, Sami & Pietikäinen, Matti. (1997).
Adaptive Document Binarization..
Pattern Recognition. 33. 147-152 vol.1. 10.1109/ICDAR.1997.619831.

From:
https://www.researchgate.net/publication/3710586_Adaptive_Document_Binarization
---------------------------------------------------------------------------------------
Official citation for Phansalkar binarization:
Neerad Phansalkar, Sumit More, Ashish Sabale and Madhuri Joshi,
"Adaptive local thresholding for detection of nuclei in diversity stained cytology
images,"
2011 International Conference on Communications and Signal Processing, Kerala, India,
2011, pp. 218-220, doi: 10.1109/ICCSP.2011.5739305.
keywords: {
Image segmentation;
Pixel;
cytology;
image segmentation;
local thresholding;
color spaces
},

From:
https://ieeexplore.ieee.org/abstract/document/5739305
---------------------------------------------------------------------------------------
Official citation for WAN binarization:
Mustafa, Wan & Kader, Mohamed. (2018).
Binarization of Document Image Using Optimum Threshold Modification.
Journal of Physics: Conference Series. 1019. 012022. 10.1088/1742-6596/1019/1/012022.

From:
https://www.researchgate.net/publication/326026836
---------------------------------------------------------------------------------------
Official citation for WOLF binarization:
Christian Wolf, Jean-Michel Jolion.
Extraction and Recognition of Artificial Text in Multimedia Documents.
Pattern Analysis and Applications, 2004, 4, 6, pp.309-326. ⟨10.1007/s10044-003-0197-7⟩.
⟨hal-01504401⟩

From:
https://hal.science/hal-01504401v1
---------------------------------------------------------------------------------------
Official citation for NICK binarization:
Khurshid, Khurram & Siddiqi, Imran & Faure, Claudie & Vincent, Nicole. (2009).
Comparison of Niblack inspired Binarization Methods for Ancient Documents.
1-10. 10.1117/12.805827.

From:
https://www.researchgate.net/publication/221253803
---------------------------------------------------------------------------------------
Official citation for Singh binarization:
Taiyenjam, Romen & Roy, Sudipta & Imocha Singh, Oinam & Sinam, Tejmani & Singh,
Khumanthem. (2012).
A New Local Adaptive Thresholding Technique in Binarization.
CoRR. abs/1201.5227. 10.48550/arXiv.1201.5227.

From:
https://arxiv.org/abs/1201.5227
---------------------------------------------------------------------------------------
"""

import numpy as np
from numpy.typing import NDArray

from otary.image.utils.local import max_local, mean_local
from otary.image.utils.tools import check_transform_window_size


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
    # pylint: disable=too-many-arguments, too-many-positional-arguments
    # pylint: disable=too-many-locals
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
