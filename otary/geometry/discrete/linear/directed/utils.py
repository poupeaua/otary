"""
Utils for the analysis of directed lines
Code directly extracted from MetPy
https://unidata.github.io/MetPy/latest/api/generated/metpy.calc.angle_to_direction.html#metpy.calc.angle_to_direction # pylint: disable=line-too-long
"""

from operator import itemgetter

import numpy as np
import pint


def setup_registry(reg):
    """Set up a given registry with MetPy's default tweaks and settings."""
    reg.autoconvert_offset_to_baseunit = True

    # Define commonly encountered units not defined by pint
    reg.define(
        "degrees_north = degree = degrees_N = degreesN = degree_north = degree_N "
        "= degreeN"
    )
    reg.define(
        "degrees_east = degree = degrees_E = degreesE = degree_east = degree_E "
        "= degreeE"
    )
    # Enable pint's built-in matplotlib support
    reg.setup_matplotlib()

    return reg


# Make our modifications using pint's application registry--which allows us to better
# interoperate with other libraries using Pint.
units = setup_registry(pint.get_application_registry())

UND = "UND"
UND_ANGLE = -999.0
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
    UND,
]  # note the order matters!

MAX_DEGREE_ANGLE = units.Quantity(360, "degree")
BASE_DEGREE_MULTIPLIER = units.Quantity(22.5, "degree")

DIR_DICT = {dir_str: i * BASE_DEGREE_MULTIPLIER for i, dir_str in enumerate(DIR_STRS)}
DIR_DICT[UND] = units.Quantity(np.nan, "degree")


def angle_to_direction(input_angle, full=False, level=3):
    """Convert the meteorological angle to directional text.

    Works for angles greater than or equal to 360 (360 -> N | 405 -> NE)
    and rounds to the nearest angle (355 -> N | 404 -> NNE)

    Parameters
    ----------
    input_angle : float or array-like
        Angles such as 0, 25, 45, 360, 410, etc.
    full : bool
        True returns full text (South), False returns abbreviated text (S)
    level : int
        Level of detail (3 = N/NNE/NE/ENE/E... 2 = N/NE/E/SE... 1 = N/E/S/W)

    Returns
    -------
    direction
        The directional text

    Examples
    --------
    >>> from metpy.calc import angle_to_direction
    >>> from metpy.units import units
    >>> angle_to_direction(225. * units.deg)
    'SW'

    """
    try:  # strip units temporarily
        origin_units = input_angle.units
        input_angle = input_angle.m
    except AttributeError:  # no units associated
        origin_units = units.degree

    if not hasattr(input_angle, "__len__") or isinstance(input_angle, str):
        input_angle = [input_angle]
        scalar = True
    else:
        scalar = False

    np_input_angle = np.array(input_angle).astype(float)
    origshape = np_input_angle.shape
    ndarray = len(origshape) > 1
    # clean any numeric strings negatives and None does not handle strings with alphabet
    input_angle = units.Quantity(np_input_angle, origin_units)
    input_angle[input_angle < 0] = np.nan

    # Normalize between 0 - 360
    norm_angles = input_angle % MAX_DEGREE_ANGLE

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
        i * BASE_DEGREE_MULTIPLIER.m * nskip: dir_str
        for i, dir_str in enumerate(DIR_STRS[::nskip])
    }
    angle_dict[MAX_DEGREE_ANGLE.m] = "N"  # handle edge case of 360.
    angle_dict[UND_ANGLE] = UND

    # round to the nearest angles for dict lookup
    # 0.001 is subtracted so there's an equal number of dir_str from
    # np.arange(0, 360, 22.5), or else some dir_str will be preferred

    # without the 0.001, level=2 would yield:
    # ['N', 'N', 'NE', 'E', 'E', 'E', 'SE', 'S', 'S',
    #  'S', 'SW', 'W', 'W', 'W', 'NW', 'N']

    # with the -0.001, level=2 would yield:
    # ['N', 'N', 'NE', 'NE', 'E', 'E', 'SE', 'SE',
    #  'S', 'S', 'SW', 'SW', 'W', 'W', 'NW', 'NW']

    multiplier = np.round((norm_angles / BASE_DEGREE_MULTIPLIER / nskip) - 0.001).m
    round_angles = multiplier * BASE_DEGREE_MULTIPLIER.m * nskip
    round_angles[np.where(np.isnan(round_angles))] = UND_ANGLE
    if ndarray:
        round_angles = round_angles.flatten()
    dir_str_arr = itemgetter(*round_angles)(angle_dict)  # returns str or tuple

    if full:
        dir_str_arr = ",".join(dir_str_arr)
        dir_str_arr = _unabbreviate_direction(dir_str_arr)
        dir_str_arr = dir_str_arr.split(",")
        if scalar:
            return dir_str_arr[0]
        return np.array(dir_str_arr).reshape(origshape)

    if scalar:
        return dir_str_arr
    return np.array(dir_str_arr).reshape(origshape)


def _unabbreviate_direction(abb_dir_str):
    """Convert abbreviated directions to non-abbreviated direction."""
    return (
        abb_dir_str.upper()
        .replace(UND, "Undefined ")
        .replace("N", "North ")
        .replace("E", "East ")
        .replace("S", "South ")
        .replace("W", "West ")
        .replace(" ,", ",")
    ).strip()
