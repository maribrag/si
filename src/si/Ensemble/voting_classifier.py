import sys
sys.path.append("C:\\Users\\maria\\Documents\\GitHub\\si")

import numpy as np

from src.si.Data.dataset import Dataset
from src.si.metrics.accuracy import accuracy


class VotingClassifier:
    """
    Ensemble classifier that uses the majority vote to predict the class labels.

    Parameters
    ----------
    models : array-like, shape = [n_models]
        Different models for the ensemble.

    Attributes
    ----------
    """
    def __init__(self, models):
        """
        Initialize the ensemble classifier.

        Parameters
        ----------
        models: array-like, shape = [n_models]
            Different models for the ensemble.

        """
        # parameters
        self.models = models

    def fit(self, dataset: Dataset) -> 'VotingClassifier':
        """
        Fit the models according to the given training data.

        Parameters
        ----------
        dataset : Dataset
            The training data.

        Returns
        -------
        self : VotingClassifier
            The fitted model.
        """
        for model in self.models:
            model.fit(dataset)

        return self

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        dataset : Dataset
            The test data.

        Returns
        -------
        y : array-like, shape = [n_samples]
            The predicted class labels.
        """

        # helper function
        def _get_majority_vote(pred: np.ndarray) -> int:
            """
            It returns the majority vote of the given predictions

            Parameters
            ----------
            pred: np.ndarray
                The predictions to get the majority vote of

            Returns
            -------
            majority_vote: int
                The majority vote of the given predictions
            """
            # get the most common label
            labels, counts = np.unique(pred, return_counts=True)
            return labels[np.argmax(counts)]

        predictions = np.array([model.predict(dataset) for model in self.models]).transpose()
        return np.apply_along_axis(_get_majority_vote, axis=1, arr=predictions)

    def score(self, dataset: Dataset) -> float:
        """
        Returns the mean accuracy on the given test data and labels.

        Parameters
        ----------
        dataset : Dataset
            The test data.

        Returns
        -------
        score : float
            Mean accuracy
        """
        return accuracy(dataset.y, self.predict(dataset))


if __name__ == '__main__':
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier

    # Generate some random data
    np.random.seed(42)
    X_train = np.random.rand(100, 5)
    y_train = np.random.randint(0, 2, size=100)
    X_test = np.random.rand(50, 5)
    y_test = np.random.randint(0, 2, size=50)

    # Create base models
    model1 = DecisionTreeClassifier()
    model2 = KNeighborsClassifier()

    # Instantiate VotingClassifier
    voting_classifier = VotingClassifier(models=[model1, model2])

    # Create a Dataset object
    train_dataset = Dataset(X_train, y_train)
    test_dataset = Dataset(X_test, y_test)

    # Fit the VotingClassifier
    voting_classifier.fit(train_dataset)

    # Calculate and print the accuracy score
    accuracy_score = voting_classifier.score(test_dataset)
    print("Accuracy of Voting Classifier:", accuracy_score)