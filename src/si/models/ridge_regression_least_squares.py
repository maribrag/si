import sys
sys.path.append("C:\\Users\\maria\\Documents\\GitHub\\si")

import numpy as np

from src.si.Data.dataset import Dataset
from src.si.metrics.mse import mse


class RidgeRegressionLeastSquares:
    def __init__(self, l2_penalty=1, scale=True):
        self.l2_penalty = l2_penalty
        self.scale = scale
        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None

    def fit(self, X, y):
        if self.scale:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
            X = (X - self.mean) / self.std

        # Add intercept term to X
        X = np.c_[np.ones(X.shape[0]), X]

        # Compute penalty term
        penalty_matrix = self.l2_penalty * np.eye(X.shape[1])
        penalty_matrix[0, 0] = 0  # Set first element to 0 for theta_zero

        # Compute model parameters using closed-form solution
        self.theta = np.linalg.inv(X.T.dot(X) + penalty_matrix).dot(X.T).dot(y)
        self.theta_zero = self.theta[0]  # First element is theta_zero
        self.theta = self.theta[1:]  # Remaining elements are theta

    def predict(self, X):
        if self.scale:
            X = (X - self.mean) / self.std

        # Add intercept term to X
        X = np.c_[np.ones(X.shape[0]), X]

        # Compute predicted Y
        predicted_y = X.dot(np.r_[self.theta_zero, self.theta])

        return predicted_y

    def score(self, X, y):
        predicted_y = self.predict(X)
        return mse(y, predicted_y)



# This is how you can test it against sklearn to check if everything is fine

if __name__ == '__main__':
    # make a linear dataset
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    dataset_ = Dataset(X=X, y=y)

    # fit the model
    model = RidgeRegressionLeastSquares(alpha=2.0)
    model.fit(dataset_)
    print(model.theta)
    print(model.theta_zero)
  
    # compute the score
    print(model.score(dataset_))

    # compare with sklearn
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=2.0)
    # scale data
    X = (dataset_.X - np.nanmean(dataset_.X, axis=0)) / np.nanstd(dataset_.X, axis=0)
    model.fit(X, dataset_.y)
    print(model.coef_) # should be the same as theta
    print(model.intercept_) # should be the same as theta_zero
    print(mse(dataset_.y, model.predict(X)))