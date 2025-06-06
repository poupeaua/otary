"""
Perfomance evaluation
"""

import time
from functools import wraps

# ruff: noqa: F401
import numpy as np
from src.cv.ocr.output import OcrMultiOutput


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
