"""
Utils for the analysis of directed lines
Code inspired by the MetPy library but without the pint library dependency
since we do not need the unit capabilities.
https://unidata.github.io/MetPy/latest/api/generated/metpy.calc.angle_to_direction.html#metpy.calc.angle_to_direction # pylint: disable=line-too-long
"""

DIR_STRS = [
    "N",
    "NNE",
    "NE",
    "ENE",
    "E",
    "ESE",
    "SE",
    "SSE",
    "S",
    "SSW",
    "SW",
    "WSW",
    "W",
    "WNW",
    "NW",
    "NNW",
]  # note the order matters!

MAX_DEGREE_ANGLE: float = 360.0
BASE_DEGREE_MULTIPLIER: float = 22.5

DIR_DICT = {dir_str: i * BASE_DEGREE_MULTIPLIER for i, dir_str in enumerate(DIR_STRS)}


def angle_to_direction(angle: float, full: bool = False, level: int = 3) -> str:
    """Convert the angle to cardinal direction text.

    Works for angles greater than or equal to 360 (360 -> N | 405 -> NE)
    and rounds to the nearest angle (355 -> N | 404 -> NNE)

    Parameters:
        angle (float): Angles such as 0, 25, 45, 360, 410, etc.
        full (bool): True returns full text (South), False returns abbreviated text (S)
        level (int): Level of detail
            (3 = N/NNE/NE/ENE/E... 2 = N/NE/E/SE... 1 = N/E/S/W)

    Returns
        direction (str): The directional text

    Examples
    --------
    >>> angle_to_direction(225)
    'SW'
    """
    norm_angle = angle % MAX_DEGREE_ANGLE

    if level == 3:
        nskip = 1
    elif level == 2:
        nskip = 2
    elif level == 1:
        nskip = 4
    else:
        err_msg = "Level of complexity cannot be less than 1 or greater than 3!"
        raise ValueError(err_msg)

    angle_dict = {
        i * BASE_DEGREE_MULTIPLIER * nskip: dir_str
        for i, dir_str in enumerate(DIR_STRS[::nskip])
    }
    angle_dict[MAX_DEGREE_ANGLE] = "N"  # handle edge case of 360.

    # round to the nearest angles for dict lookup
    # 0.001 is subtracted so there's an equal number of dir_str from
    # np.arange(0, 360, 22.5), or else some dir_str will be preferred

    # without the 0.001, level=2 would yield:
    # ['N', 'N', 'NE', 'E', 'E', 'E', 'SE', 'S', 'S',
    #  'S', 'SW', 'W', 'W', 'W', 'NW', 'N']

    # with the -0.001, level=2 would yield:
    # ['N', 'N', 'NE', 'NE', 'E', 'E', 'SE', 'SE',
    #  'S', 'S', 'SW', 'SW', 'W', 'W', 'NW', 'NW']

    multiplier = round((norm_angle / BASE_DEGREE_MULTIPLIER / nskip) - 0.001)
    round_angle = multiplier * BASE_DEGREE_MULTIPLIER * nskip

    direction = angle_dict[round_angle]

    if full:
        direction = ",".join(direction)
        direction = _unabbreviate_direction(direction)
        return direction.split(",", maxsplit=1)[0]

    return direction


def _unabbreviate_direction(direction: str) -> str:
    """Convert abbreviated directions to non-abbreviated direction."""
    return (
        direction.upper()
        .replace("N", "North ")
        .replace("E", "East ")
        .replace("S", "South ")
        .replace("W", "West ")
        .replace(" ,", ",")
    ).strip()
