"""
FAIR clustering methods.

They all suit the input of the FAIR algorithm.
The FAIR algorithm always expect a clustering method with an input of shape (N, n, n)
meaning N patches of shape (n, n).
"""

from typing import Literal

from numpy.typing import NDArray

from otary.image.components.transformer.components.binarizer.ops.fair.clustering.kmeans import (
    kmeans,
)

from otary.image.components.transformer.components.binarizer.ops.fair.clustering.expectation_maximization import (
    expectation_maximization,
)

from otary.image.components.transformer.components.binarizer.ops.fair.clustering.otsu_cluster import (
    otsu_clustering,
)

FAIR_CLUSTERING_ALGORITHMS = Literal["kmeans", "em", "otsu"]


def fair_clustering(
    x: NDArray,
    tol: float = 1e-2,
    max_iter: int = 100,
    algorithm: str = "kmeans",
) -> NDArray:
    """Applies a FAIR clustering method to a batch of patches.

    Args:
        x (NDArray): input patches as shape (N, n, n)
        tol (float, optional): tolerance. Defaults to 1e-2.
        max_iter (int, optional): Maximum number of iterations to check clustering
            algorithm convergence. Defaults to 100.
        algorithm (str, optional): name of the clustering algorithm.
            Defaults to "kmeans".

    Returns:
        NDArray: output threshold as shape (N, n, n) where 0 is background and
            1 is foreground
    """
    if algorithm == "kmeans":
        return kmeans(x, max_iter, tol)
    elif algorithm == "em":
        return expectation_maximization(x, tol, max_iter)
    elif algorithm == "otsu":
        return otsu_clustering(x)
    else:
        raise ValueError(
            f"Unknown FAIR clustering algorithm: {algorithm}. "
            f"Available algorithms: {FAIR_CLUSTERING_ALGORITHMS}. "
            "em stands for Expectation Maximization."
        )
