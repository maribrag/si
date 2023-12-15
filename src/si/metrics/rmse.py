import sys
sys.path.append("C:\\Users\\maria\\Documents\\GitHub\\si")

import numpy as np

def rmse(y_true, y_pred):
    """
    Calculate the Root Mean Squared Error (RMSE) between y_true and y_pred.

    Parameters
    ----------
    y_true : array-like
        Real values of y.
    y_pred : array-like
        Predicted values of y.

    Returns
    -------
    float
        RMSE between y_true and y_pred.
    """
    squared_error = np.square(np.subtract(y_true, y_pred))
    mean_squared_error = np.mean(squared_error)
    root_mean_squared_error = np.sqrt(mean_squared_error)
    return root_mean_squared_error
