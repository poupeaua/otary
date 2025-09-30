"""
Clustering K-Means algorithm
"""

from typing import Optional

import numpy as np
from numpy.typing import NDArray


def kmeans_1d(
    x: NDArray,
    max_iter: int = 100,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
) -> NDArray:
    """Performs K-means clustering for 2 clusters on a 1D array.

    Args:
        X (np.ndarray): Input 1D array of data points.
        max_iter (int, optional): Maximum number of iterations. Defaults to 100.
        tol (float, optional): Tolerance for convergence. Defaults to 1e-4.
        random_state (int, optional): Seed for random number generator.
            Defaults to None.

    Returns:
        labels (NDArray): Array of cluster assignments for each data point.
    """
    if random_state is not None:
        np.random.seed(random_state)

    x = x.reshape(-1)

    centroids = np.array([0, 255])

    for _ in range(max_iter):
        distances = np.abs(x[:, None] - centroids[None, :])
        labels = np.argmin(distances, axis=1)

        new_centroids = np.array(
            [
                x[labels == j].mean() if np.any(labels == j) else centroids[j]
                for j in range(2)
            ]
        )

        converged = np.all(np.abs(new_centroids - centroids) < tol)
        if converged:
            break
        centroids = new_centroids

    # if label 1 has higher mean (so brighter pixels) flip everything
    if centroids[0] < centroids[1]:
        labels = 1 - labels
        centroids = centroids[::-1]

    return labels


def kmeans(
    x: NDArray,
    max_iter: int = 100,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
) -> NDArray:
    """Applies 1D K-means clusteringfor 2 clusters to each (n, n) patch in a batch of
    data.

    Args:
        patches (np.ndarray): Array of shape (N, n, n), where each entry is a
            patch of size (n, n)
        max_iter (int, optional): Maximum number of iterations. Defaults to 100.
        tol (float, optional): Tolerance for convergence. Defaults to 1e-4.
        random_state (int, optional): Seed for random number generator.
            Defaults to None.

    Returns:
        labels_list (NDArray): List of label maps of shape (n, n) for
            each patch.
    """
    assert x.ndim == 3
    labels_list = []
    for patch in x:
        labels = kmeans_1d(patch.flatten(), max_iter, tol, random_state)
        labels_list.append(labels.reshape(patch.shape))  # reshape back to (n, n)

    return np.asarray(labels_list)
