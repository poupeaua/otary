"""
All thresholding / binarization methods
"""

__all__ = [
    "threshold_niblack_like",
    "threshold_isauvola",
    "threshold_gatos",
    "threshold_su",
    "threshold_bernsen",
    "threshold_bradley_roth",
    "threshold_feng",
    "threshold_adotsu",
    "threshold_fair",
]

from otary.image.components.transformer.components.binarizer.ops.niblack_like import (
    threshold_niblack_like,
)
from otary.image.components.transformer.components.binarizer.ops.isauvola import (
    threshold_isauvola,
)
from otary.image.components.transformer.components.binarizer.ops.gatos import (
    threshold_gatos,
)
from otary.image.components.transformer.components.binarizer.ops.su import (
    threshold_su,
)
from otary.image.components.transformer.components.binarizer.ops.bernsen import (
    threshold_bernsen,
)
from otary.image.components.transformer.components.binarizer.ops.bradley_roth import (
    threshold_bradley_roth,
)
from otary.image.components.transformer.components.binarizer.ops.feng import (
    threshold_feng,
)
from otary.image.components.transformer.components.binarizer.ops.adotsu import (
    threshold_adotsu,
)
from otary.image.components.transformer.components.binarizer.ops.fair.fair import (
    threshold_fair,
)
