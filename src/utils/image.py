"""
File utils for image manipulation
"""

import cv2
import scipy.ndimage
import numpy as np
import matplotlib.pyplot as plt
from src.geometry.segment import Segment
from src.utils.geometry import compute_slope_angle

def show_image(image: np.ndarray, title=None, conversion=cv2.COLOR_BGR2RGB):
    # Converts from one colour space to the other. this is needed as RGB
    # is not the default colour space for OpenCV
    image = image.copy()
    
    if conversion is not None:
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
    lines = lines.astype(int)
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
    """Show the image and the bounding boxes from the OCR output.
    It allows you to show bounding boxes that can have an angle, not necessarily vertical or
    horizontal.

    Args:
        image (np.ndarray): _description_
        ocr_output (np.ndarray): _description_
        title (_type_, optional): _description_. Defaults to None.
        conversion (_type_, optional): _description_. Defaults to cv2.COLOR_BGR2RGB.
        default_bbox_color (tuple, optional): _description_. Defaults to (125, 125, 230).
    """
    _img = image.copy()
    _img = cv2.cvtColor(_img, cv2.COLOR_GRAY2RGB)
    for o in ocr_output:
        bbox = np.array(o[0])
        text = o[1]
        confidence_level = o[2]
        cnt = [np.array(bbox).reshape((-1,1,2)).astype(np.int32)]
        _img = cv2.drawContours(_img, contours=cnt, contourIdx=-1, thickness=2, color=default_bbox_color)
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
        resize_scale_percent: int=100,
        extra_length_default_width: int=75,
        show_translation: bool=False
    ):
    _img = image.copy()
    
    # center the image based on the middle of the line
    center_pt_img = (np.array([_img.shape[1], _img.shape[0]]) / 2).astype(int)
    _img = center_image_to_line(image=_img, line=line)
    
    if show_translation:
        middle_pt_line = (np.sum(line, axis=0) / 2).astype(int)
        _img2 = cv2.cvtColor(_img, cv2.COLOR_GRAY2RGB)
        cv2.circle(_img2, center=middle_pt_line, radius=2, color=(0, 0, 255), thickness=5)
        cv2.circle(_img2, center=center_pt_img, radius=2, color=(0, 0, 255), thickness=5)
        cv2.arrowedLine(_img2, pt1=middle_pt_line, pt2=center_pt_img, color=(0, 0, 255), thickness=5)
        show_image(_img2)

    # rotate the image so that the line is horizontal
    angle_degree = compute_slope_angle(line, degree=True)
    _img = scipy.ndimage.rotate(input=_img, angle=angle_degree, reshape=True)

    # crop the image
    center_pt_img = (np.array([_img.shape[1], _img.shape[0]]) / 2).astype(int)
    if width_crop_rect is None:
        # default the width for crop to be a bit more than line length
        width_crop_rect = Segment(line).length + extra_length_default_width
    x0, y0 = int(center_pt_img[1] - heigth_crop_rect / 2), int(center_pt_img[0] - width_crop_rect / 2)
    x1, y1 = int(center_pt_img[1] + heigth_crop_rect / 2), int(center_pt_img[0] + width_crop_rect / 2)
    _img_cropped = _img[x0:x1, y0:y1]

    _img_cropped_resized = resize_image(image=_img_cropped, scale_percent=resize_scale_percent)
    return _img_cropped_resized
