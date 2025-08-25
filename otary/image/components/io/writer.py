"""
WriterImage module
"""

from PIL import Image as ImagePIL

from otary.image.base import BaseImage


class WriterImage:
    """WriterImage class that provide methods to save and show the image"""

    def __init__(self, base: BaseImage) -> None:
        self.base = base

    def show(
        self,
        figsize: tuple[float, float] = (-1, -1),
        popup_window_display: bool = False,
    ) -> ImagePIL.Image:
        """Show the image

        Args:
            figsize (tuple[float, float], optional): size of the figure.
                Defaults to (-1, -1), meaning the original size of the image.
            popup_window_display (bool, optional): whether to display the image in a
                popup window. Defaults to False.
        """
        if figsize[0] <= 0 and figsize[1] <= 0:
            figsize = (self.base.width, self.base.height)
        else:
            if figsize[1] <= 0:
                aspect_ratio = self.base.height / self.base.width
                figsize = (figsize[0], figsize[0] * aspect_ratio)

            elif figsize[0] <= 0:
                aspect_ratio = self.base.width / self.base.height
                figsize = (figsize[1] * aspect_ratio, figsize[1])

        figsize = (int(figsize[0]), int(figsize[1]))

        self.base.as_reversed_color_channel()
        im = self.base.as_pil().resize(size=figsize)

        if popup_window_display:
            im.show()

        return im

    def save(self, fp: str) -> None:
        """Save the image in a local file

        Args:
            fp (str): fp stands for filepath which is the path to the file
        """
        self.base.as_reversed_color_channel().as_pil().save(fp)
