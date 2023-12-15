
import numpy as np

def manhattan_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Calculates the Manhattan distance between a single sample x and multiple samples y.

    Parameters
    ----------
    x : np.ndarray
        Single sample.
    y : np.ndarray
        Multiple samples.

    Returns
    -------
    np.ndarray
        Array containing the distances between x and various samples in y.
    """
    return np.sum(np.abs(x - y), axis=1)
