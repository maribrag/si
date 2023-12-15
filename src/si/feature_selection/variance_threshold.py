import sys
sys.path.append("C:\\Users\\maria\\Documents\\GitHub\\si")
import numpy as np

from src.si.Data.dataset import Dataset


class VarianceThreshold:
    """
    Variance Threshold feature selection.
    Features with a training-set variance lower than this threshold will be removed from the dataset.

    Parameters
    ----------
    threshold: float
        The threshold value to use for feature selection. Features with a
        training-set variance lower than this threshold will be removed.

    Attributes
    ----------
    variance: array-like, shape (n_features,)
        The variance of each feature.
    """

    def __init__(self, threshold: float = 0.0):
        """
        Variance Threshold feature selection.
        Features with a training-set variance lower than this threshold will be removed from the dataset.

        Parameters
        ----------
        threshold: float
            The threshold value to use for feature selection. Features with a
            training-set variance lower than this threshold will be removed.
        """
        if threshold < 0:
            raise ValueError("Threshold must be non-negative")

        # parameters
        self.threshold = threshold

        # attributes
        self.variance = None

    def fit(self, dataset: Dataset) -> 'VarianceThreshold':
        """
        Fit the VarianceThreshold model according to the given training data.
        Parameters
        ----------
        dataset : Dataset
            The dataset to fit.

        Returns
        -------
        self : object
        """
        self.variance = np.var(dataset.X, axis=0)
        return self

    def transform(self, dataset: Dataset) -> Dataset:
        """
        It removes all features whose variance does not meet the threshold.
        Parameters
        ----------
        dataset: Dataset

        Returns
        -------
        dataset: Dataset
        """
        X = dataset.X

        features_mask = self.variance > self.threshold
        X = X[:, features_mask]
        features = np.array(dataset.features)[features_mask]
        return Dataset(X=X, y=dataset.y, features=list(features), label=dataset.label)

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        Fit to data, then transform it.
        Parameters
        ----------
        dataset: Dataset

        Returns
        -------
        dataset: Dataset
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
    
    # Instantiate VarianceThreshold with a threshold
    variance_threshold = VarianceThreshold(threshold=0.1)
    
    # Fit and transform the dataset
    transformed_dataset = variance_threshold.fit_transform(dataset)
    
    # Display the transformed dataset
    print("Original Features:", dataset.features)
    print("Selected Features:", transformed_dataset.features)
    print("Transformed X shape:", transformed_dataset.X.shape)