import sys
sys.path.append("C:\\Users\\maria\\Documents\\GitHub\\si")
import numpy as np
from src.si.Data.dataset import Dataset
from src.si.Statistics.f_classification import f_classification

class SelectPercentile:
    def __init__(self, score_func=f_classification, percentile=10):
        self.score_func = score_func
        self.percentile = percentile
        self.F = None
        self.p = None

    def fit(self, dataset):
        self.F, self.p = self.score_func(dataset)
        return self

    def transform(self, dataset):
        n_features = dataset.X.shape[1]
        k = int(n_features * (self.percentile / 100))
        idxs = self.F.argsort()[-k:][::-1]
        selected_features = dataset.X[:, idxs]
        selected_feature_names = [dataset.features[i] for i in idxs]
        return Dataset(selected_features, dataset.y, selected_feature_names, dataset.label)

    def fit_transform(self, dataset):
        self.fit(dataset)
        return self.transform(dataset)

if __name__ == '__main__':
    # Generate some random data
    np.random.seed(42)
    X = np.random.rand(100, 10)  # 100 samples with 10 features
    y = np.random.randint(0, 2, size=100)  # binary labels
    
    # Create a Dataset object
    dataset = Dataset(X, y)
    
    # Instantiate SelectPercentile with default parameters
    select_percentile = SelectPercentile()
    
    # Fit and transform the dataset
    transformed_dataset = select_percentile.fit_transform(dataset)
    
    # Display the transformed dataset
    print("Original Features:", dataset.features)
    print("Selected Features:", transformed_dataset.features)
    print("Transformed X shape:", transformed_dataset.X.shape)