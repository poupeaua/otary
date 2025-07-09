from typing import Optional

import cv2
import matplotlib.pyplot as plt

from otary.image.base import BaseImage


class WriterImage:

    def __init__(self, base: BaseImage) -> None:
        self.base = base

    def show(
        self,
        title: Optional[str] = None,
        figsize: tuple[float, float] = (8.0, 6.0),
        color_conversion: int = cv2.COLOR_BGR2RGB,
        save_filepath: Optional[str] = None,
    ) -> None:
        """Show the image

        Args:
            title (Optional[str], optional): title of the image. Defaults to None.
            figsize (tuple[float, float], optional): size of the figure.
                Defaults to (8.0, 6.0).
            color_conversion (int, optional): color conversion parameter.
                Defaults to cv2.COLOR_BGR2RGB.
            save_filepath (Optional[str], optional): save the image if needed.
                Defaults to None.
        """
        # Converts from one colour space to the other. this is needed as RGB
        # is not the default colour space for OpenCV
        if color_conversion is not None:
            im = cv2.cvtColor(self.base.asarray, color_conversion)
        else:
            im = self.base.asarray

        plt.figure(figsize=figsize)

        # Show the image
        plt.imshow(im)

        # remove the axis / ticks for a clean looking image
        plt.xticks([])
        plt.yticks([])

        # if a title is provided, show it
        if title is not None:
            plt.title(title)

        if save_filepath is not None:
            plt.savefig(save_filepath)

        plt.show()

    def save(self, save_filepath: str) -> None:
        """Save the image in a local file

        Args:
            save_filepath (str): path to the file
        """
        self.show(save_filepath=save_filepath)
