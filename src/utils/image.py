"""
File utils for image manipulation
"""

import cv2
import scipy.ndimage
import numpy as np
import matplotlib.pyplot as plt
from src.utils.geometry import compute_slope_angle

def show_image(image: np.ndarray, title=None, conversion=cv2.COLOR_BGR2RGB):
    # Converts from one colour space to the other. this is needed as RGB
    # is not the default colour space for OpenCV
    image = cv2.cvtColor(image, conversion)

    # Show the image
    plt.imshow(image)

    # remove the axis / ticks for a clean looking image
    plt.xticks([])
    plt.yticks([])

    # if a title is provided, show it
    if title is not None:
        plt.title(title)

    plt.show()
    
def show_image_and_lines(
        image: np.ndarray, 
        lines: np.ndarray,
        colors_lines: np.ndarray=None,
        title=None, 
        conversion=cv2.COLOR_BGR2RGB,
        default_color = (0, 255, 255)
    ):
    if colors_lines is None:
        colors_lines = [default_color for i in range(len(lines))]
    assert len(lines) == len(colors_lines)
    
    img_copy = image.copy()
    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2BGR)
    for line, color in zip(lines, colors_lines):
        cv2.line(img_copy, line[0], line[1], color, 3, cv2.LINE_AA)
    show_image(image=img_copy, title=title, conversion=conversion)
    
    
def show_image_ocr(
        image: np.ndarray,
        ocr_output: np.ndarray,
        title=None, 
        conversion=cv2.COLOR_BGR2RGB,
        default_bbox_color=(125, 125, 230)
    ):
    _img = image.copy()
    _img = cv2.cvtColor(_img, cv2.COLOR_GRAY2RGB)
    for o in ocr_output:
        bbox = np.array(o[0])
        text = o[1]
        confidence_level = o[2]
        sums_coord = bbox.sum(axis=1)
        start_point, end_point = np.array(bbox[np.argmin(sums_coord)], dtype=int), \
                                np.array(bbox[np.argmax(sums_coord)], dtype=int)
        _img = cv2.rectangle(img=_img, pt1=start_point, pt2=end_point, thickness=2, color=default_bbox_color)
    show_image(image=_img, title=title, conversion=conversion)
    
def center_image_to_point(image: np.ndarray, point: np.ndarray, mode: str="constant"):
    """Shift the image so that the input point ends up in the middle of the new image

    Args:
        image (np.ndarray): shape (lx, ly)
        point (np.ndarray): shape (2,)
        mode (str, optional): _description_. Defaults to "constant".

    Returns:
        (np.ndarray): shape (lx, ly)
    """
    center_img_vector = (np.array([image.shape[1], image.shape[0]]) / 2).astype(int)
    shift_vector = center_img_vector - point
    img = scipy.ndimage.shift(input=image, shift=np.roll(shift_vector, 1), mode=mode)
    return img
    
def center_image_to_line(image: np.ndarray, line: np.ndarray, mode: str="constant"):
    """Shift the image so that the line middle point ends up in the middle of the new image

    Args:
        image (np.ndarray): shape (lx, ly)
        point (np.ndarray): shape (2,)
        mode (str, optional): _description_. Defaults to "constant".

    Returns:
        _type_: _description_
    """
    point_center_line = (np.sum(line, axis=0) / 2).astype(int)
    return center_image_to_point(image=image, point=point_center_line)

def resize_image(image: np.ndarray, scale_percent: float):
    """Resize the image to a new size

    Args:
        image (np.ndarray): (lx, ly)
        scale_percent (float): scale to resize the image. A value 100 does not change the image.
            200 double the image size.

    Returns:
        (np.ndarray): new resized image
    """
    img = image.copy()
    if scale_percent == 100:
        return img
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return img
    
    
def crop_image_around_line_horizontal(
        image: np.ndarray, 
        line: np.ndarray,
        heigth_crop_rect: int=100,
        width_crop_rect: int=None,
        scale_percent: int=100,
        extra_length_default_width: int=75,
        show_central_points: bool=False
    ):
    _img = image.copy()
    
    # center the image based on the middle of the line
    center_pt_img = (np.array([_img.shape[1], _img.shape[0]]) / 2).astype(int)
    _img = center_image_to_line(image=_img, line=line)
    
    if show_central_points:
        middle_pt_line = (np.sum(line, axis=0) / 2).astype(int)
        cv2.circle(_img, center=middle_pt_line, radius=2, color=(0, 0, 255), thickness=5)
        cv2.circle(_img, center=center_pt_img, radius=2, color=(0, 0, 255), thickness=5)

    # rotate the image so that the line is horizontal
    angle_degree = compute_slope_angle(line, degree=True)
    _img = scipy.ndimage.rotate(input=_img, angle=angle_degree, reshape=False)

    # crop the image
    if width_crop_rect is None:
        # default the width for crop to be a bit more than line length
        width_crop_rect = np.linalg.norm(np.diff(line, axis=0)) + extra_length_default_width
    x0, y0 = int(center_pt_img[1] - heigth_crop_rect / 2), int(center_pt_img[0] - width_crop_rect / 2)
    x1, y1 = int(center_pt_img[1] + heigth_crop_rect / 2), int(center_pt_img[0] + width_crop_rect / 2)
    _img_cropped = _img[x0:x1, y0:y1]

    _img_cropped_resized = resize_image(image=_img_cropped, scale_percent=scale_percent)
    return _img_cropped_resized
