import sys
sys.path.append("C:\\Users\\maria\\Documents\\GitHub\\si")

from src.si.Data.dataset import Dataset
from typing import List, Tuple
import numpy as np
from src.si.models.decision_tree_classifier import DecisionTreeClassifier  

class RandomForestClassifier:
    def __init__(self, n_estimators: int, max_features: int = None, min_sample_split: int = 2,
                 max_depth: int = 10, mode: str = 'gini', seed: int = None):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.mode = mode
        self.seed = seed
        self.trees = []

    def fit(self, dataset: Dataset) -> 'RandomForestClassifier':
        np.random.seed(self.seed)

        if self.max_features is None:
            self.max_features = int(np.sqrt(dataset.X.shape[1]))

        for _ in range(self.n_estimators):
            bootstrap_indices = np.random.choice(dataset.X.shape[0], dataset.X.shape[0], replace=True)
            feature_indices = np.random.choice(dataset.X.shape[1], self.max_features, replace=False)

            bootstrap_dataset = Dataset(dataset.X[bootstrap_indices][:, feature_indices], dataset.y[bootstrap_indices])

            tree = DecisionTreeClassifier(min_sample_split=self.min_sample_split,
                                          max_depth=self.max_depth,
                                          mode=self.mode).fit(bootstrap_dataset)

            self.trees.append((feature_indices, tree))

        return self

    def predict(self, dataset: Dataset) -> np.ndarray:
        predictions = []
        for sample in dataset.X:
            tree_predictions = [tree.predict_single(sample[feature_indices]) for feature_indices, tree in self.trees]
            most_common_prediction = max(set(tree_predictions), key=tree_predictions.count)
            predictions.append(most_common_prediction)

        return np.array(predictions)

    def score(self, dataset: Dataset) -> float:
        predicted = self.predict(dataset)
        accuracy = np.mean(predicted == dataset.y)
        return accuracy
    
if __name__ == '__main__':
    # Create a sample dataset
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, size=100)
    dataset = Dataset(X, y)

    # Instantiate RandomForestClassifier
    random_forest = RandomForestClassifier(n_estimators=10, max_depth=5, seed=42)

    # Fit the model
    random_forest.fit(dataset)

    # Evaluate model on the same dataset
    accuracy = random_forest.score(dataset)
    print(f"Accuracy on the dataset: {accuracy}")
