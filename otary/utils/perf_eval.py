"""
Perfomance evaluation
"""

# ruff: noqa: F401
# pylint: disable=unused-import
import pprint
import time
from functools import wraps

import numpy as np
from otary.utils.cv.ocrsingleoutput import OcrSingleOutput


def timer(func):
    """Timer decorator to quickly measure the execution time of a function"""

    @wraps(func)
    def wrapper_timer(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        elapsed = end - start
        print(f"[TIMER] '{func.__name__}' executed in {elapsed:.4f} seconds")
        return result

    return wrapper_timer
