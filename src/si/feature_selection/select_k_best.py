import sys
sys.path.append("C:\\Users\\maria\\Documents\\GitHub\\si")

from typing import Callable

import numpy as np

from src.si.Data.dataset import Dataset
from src.si.Statistics.f_classification import f_classification


class SelectKBest:
    """
    Select features according to the k highest scores.
    Feature ranking is performed by computing the scores of each feature using a scoring function:
        - f_classification: ANOVA F-value between label/feature for classification tasks.
        - f_regression: F-value obtained from F-value of r's pearson correlation coefficients for regression tasks.

    Parameters
    ----------
    score_func: callable
        Function taking dataset and returning a pair of arrays (scores, p_values)
    k: int, default=10
        Number of top features to select.

    Attributes
    ----------
    F: array, shape (n_features,)
        F scores of features.
    p: array, shape (n_features,)
        p-values of F-scores.
    """
    def __init__(self, score_func: Callable = f_classification, k: int = 10):
        """
        Select features according to the k highest scores.

        Parameters
        ----------
        score_func: callable
            Function taking dataset and returning a pair of arrays (scores, p_values)
        k: int, default=10
            Number of top features to select.
        """
        self.k = k
        self.score_func = score_func
        self.F = None
        self.p = None

    def fit(self, dataset: Dataset) -> 'SelectKBest':
        """
        It fits SelectKBest to compute the F scores and p-values.

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset

        Returns
        -------
        self: object
            Returns self.
        """
        self.F, self.p = self.score_func(dataset)
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        """
        It transforms the dataset by selecting the k highest scoring features.

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset

        Returns
        -------
        dataset: Dataset
            A labeled dataset with the k highest scoring features.
        """
        idxs = np.argsort(self.F)[-self.k:]
        features = np.array(dataset.features)[idxs]
        return Dataset(X=dataset.X[:, idxs], y=dataset.y, features=list(features), label=dataset.label)

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        It fits SelectKBest and transforms the dataset by selecting the k highest scoring features.

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset

        Returns
        -------
        dataset: Dataset
            A labeled dataset with the k highest scoring features.
        """
        self.fit(dataset)
        return self.transform(dataset)

if __name__ == '__main__':
    # Generate some random data
    np.random.seed(42)
    X = np.random.rand(100, 10)  # 100 samples with 10 features
    y = np.random.randint(0, 2, size=100)  # binary labels
    
    # Create a Dataset object
    dataset = Dataset(X, y)
    
    # Instantiate SelectKBest with default parameters
    select_k_best = SelectKBest()
    
    # Fit and transform the dataset
    transformed_dataset = select_k_best.fit_transform(dataset)
    
    # Display the transformed dataset
    print("Original Features:", dataset.features)
    print("Selected Features:", transformed_dataset.features)
    print("Transformed X shape:", transformed_dataset.X.shape)
