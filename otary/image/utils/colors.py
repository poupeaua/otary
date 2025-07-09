"""
Utils file related to color operations
"""


def interpolate_color(
    alpha: float,
    as_bgr: bool = False,
    start_color: tuple = (255, 0, 0),
    mid_color: tuple = (255, 255, 0),
    end_color: tuple = (0, 255, 0),
) -> tuple:
    """Interpolates between start_color, mid_color, and end_color
    based on the parameter alpha.

    By default, it interpolates from red to yellow to green.
    This is the default behaviour as it is mostly used to show red for a low score (0),
    yellow for a medium score (0.5), and green for a high score (1).

    Args:
        alpha (float): Parameter ranging from 0 to 1.
            0 corresponds to start_color, 0.5 to mid_color, and 1 to end_color.
        as_bgr (bool, optional): Whether to return a BGR color or RGB color.
            Defaults to False.
        start_color (tuple, optional): Color at alpha=0. Defaults to (255, 0, 0).
        mid_color (tuple, optional): Color at alpha=0.5. Defaults to (255, 255, 0).
        end_color (tuple, optional): Color at alpha=1. Defaults to (0, 255, 0).

    Returns:
        tuple: RGB color as a tuple (R, G, B) where each value is in the range [0, 255].
    """
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")

    if alpha <= 0.5:
        # Interpolate between start_color and mid_color
        t = alpha * 2
        r = int((1 - t) * start_color[0] + t * mid_color[0])
        g = int((1 - t) * start_color[1] + t * mid_color[1])
        b = int((1 - t) * start_color[2] + t * mid_color[2])
    else:
        # Interpolate between mid_color and end_color
        t = (alpha - 0.5) * 2
        r = int((1 - t) * mid_color[0] + t * end_color[0])
        g = int((1 - t) * mid_color[1] + t * end_color[1])
        b = int((1 - t) * mid_color[2] + t * end_color[2])

    if as_bgr:
        return (b, g, r)
    return (r, g, b)
