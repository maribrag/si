import sys
sys.path.append("C:\\Users\\maria\\Documents\\GitHub\\si")

import numpy as np
from src.si.Statistics import euclidean_distance
class KNNRegressor:
    def __init__(self, k=1, distance= euclidean_distance):
        self.k = k
        self.distance = distance
        self.dataset = None

    def fit(self, dataset):
        self.dataset = dataset
        return self

    def predict(self, test_dataset):
        predictions = []
        for sample in test_dataset.X:
            distances = self.distance(sample, self.dataset.X)
            k_nearest_indices = np.argsort(distances)[:self.k]
            k_nearest_values = self.dataset.y[k_nearest_indices]
            predicted_value = np.mean(k_nearest_values)
            predictions.append(predicted_value)
        return np.array(predictions)

    def score(self, test_dataset):
        predictions = self.predict(test_dataset)
        actual_values = test_dataset.y
        rmse = np.sqrt(np.mean(np.square(predictions - actual_values)))
        return rmse



