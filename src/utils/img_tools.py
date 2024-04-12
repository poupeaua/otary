"""
File utils for image manipulation
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_rgb_image(image: np.array, title=None, conversion=cv2.COLOR_BGR2RGB):
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
        image: np.array, 
        lines: np.array,
        colors_lines: np.array=None,
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
    show_rgb_image(image=img_copy, title=title, conversion=conversion)