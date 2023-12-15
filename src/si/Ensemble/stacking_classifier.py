import sys
sys.path.append("C:\\Users\\maria\\Documents\\GitHub\\si")

from src.si.metrics.accuracy import accuracy
import numpy as np
from typing import List

class StackingClassifier:
    def __init__(self, models: List, final_model):
        self.models = models
        self.final_model = final_model
        self.predictions = []

    def fit(self, X_train, y_train):
        # Train the initial set of models
        for model in self.models:
            model.fit(X_train, y_train)
            # Get predictions from the initial set of models
            predictions = model.predict(X_train)
            self.predictions.append(predictions)

        # Train the final model with the predictions of the initial set of models
        # Assuming final_model has a fit method
        # Here, we'll concatenate the predictions horizontally to use as features for the final model
        # You may need to adjust this depending on the final model and how you want to use predictions
        final_features = np.hstack(self.predictions)
        self.final_model.fit(final_features, y_train)

        return self

    def predict(self, X_test):
        test_predictions = []
        # Get predictions from the initial set of models on the test data
        for model in self.models:
            test_predictions.append(model.predict(X_test))

        # Get the final predictions using the final model and the predictions of the initial set of models
        # Assuming final_model has a predict method
        # Here, we'll concatenate the test predictions horizontally
        final_test_features = np.hstack(test_predictions)
        return self.final_model.predict(final_test_features)

    def score(self, X_test, y_test):
        # Get predictions using the predict method
        predictions = self.predict(X_test)
        # Compute the accuracy between predicted and real values
        return accuracy(y_test, predictions)
    
if __name__ == '__main__':
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier

    # Generate some random data
    np.random.seed(42)
    X_train = np.random.rand(100, 5)
    y_train = np.random.randint(0, 2, size=100)
    X_test = np.random.rand(50, 5)
    y_test = np.random.randint(0, 2, size=50)

    # Create base models
    model1 = DecisionTreeClassifier()
    model2 = RandomForestClassifier()

    # Create final model
    final_model = DecisionTreeClassifier()

    # Instantiate StackingClassifier
    stacking_classifier = StackingClassifier(models=[model1, model2], final_model=final_model)

    # Fit the StackingClassifier
    stacking_classifier.fit(X_train, y_train)

    # Calculate and print the accuracy score
    accuracy_score = stacking_classifier.score(X_test, y_test)
    print("Accuracy of Stacking Classifier:", accuracy_score)   