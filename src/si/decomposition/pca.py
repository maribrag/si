import sys
sys.path.append("C:\\Users\\maria\\Documents\\GitHub\\si")

import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None

    def fit(self, X):
        # Calculate the mean of the samples
        self.mean = np.mean(X, axis=0)
        
        # Center the data by subtracting the mean
        centered_data = X - self.mean
        
        # Calculate the covariance matrix
        cov_matrix = np.cov(centered_data, rowvar=False)
        
        # Calculate the eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # Sort eigenvectors based on eigenvalues
        sorted_indices = np.argsort(eigenvalues)[::-1]
        self.components = eigenvectors[:, sorted_indices[:self.n_components]]
        self.explained_variance = np.diag(eigenvalues[sorted_indices[:self.n_components]])

    def transform(self, X):
        # Center the data
        centered_data = X - self.mean
        
        # Project the data onto the principal components
        transformed_data = np.dot(centered_data, self.components)
        
        return transformed_data

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

import numpy as np
from sklearn.decomposition import PCA as SKPCA

if __name__ == '__main__':

    # Generating a random dataset
    np.random.seed(0)
    X = np.random.rand(100, 4)  # 100 samples with 4 features

    # Using the custom PCA
    pca_custom = PCA(n_components=2)
    reduced_data_custom = pca_custom.fit_transform(X)

    # Using PCA from scikit-learn
    pca_sklearn = SKPCA(n_components=2)
    reduced_data_sklearn = pca_sklearn.fit_transform(X)

    # Compare results
    print("Reduced Data (Custom PCA):\n", reduced_data_custom)
    print("\nReduced Data (PCA from scikit-learn):\n", reduced_data_sklearn)