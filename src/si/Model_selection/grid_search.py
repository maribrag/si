import sys
sys.path.append("C:\\Users\\maria\\Documents\\GitHub\\si")

import itertools
from typing import Callable, Tuple, Dict, Any

import numpy as np

from src.si.Data.dataset import Dataset
from src.si.Model_selection.cross_validation import k_fold_cross_validation


def grid_search_cv(model,
                   dataset: Dataset,
                   hyperparameter_grid: Dict[str, Tuple],
                   scoring: Callable = None,
                   cv: int = 5) -> Dict[str, Any]:
    """
    Performs a grid search cross validation on a model.

    Parameters
    ----------
    model
        The model to cross validate.
    dataset: Dataset
        The dataset to cross validate on.
    hyperparameter_grid: Dict[str, Tuple]
        The hyperparameter grid to use.
    scoring: Callable
        The scoring function to use.
    cv: int
        The cross validation folds.

    Returns
    -------
    results: Dict[str, Any]
        The results of the grid search cross validation. Includes the scores, hyperparameters,
        best hyperparameters and best score.
    """
    # validate the parameter grid
    for parameter in hyperparameter_grid:
        if not hasattr(model, parameter):
            raise AttributeError(f"Model {model} does not have parameter {parameter}.")

    results = {'scores': [], 'hyperparameters': []}

    # for each combination
    for combination in itertools.product(*hyperparameter_grid.values()):

        # parameter configuration
        parameters = {}

        # set the parameters
        for parameter, value in zip(hyperparameter_grid.keys(), combination):
            setattr(model, parameter, value)
            parameters[parameter] = value

        # cross validate the model
        score = k_fold_cross_validation(model=model, dataset=dataset, scoring=scoring, cv=cv)

        # add the score
        results['scores'].append(np.mean(score))

        # add the hyperparameters
        results['hyperparameters'].append(parameters)

    results['best_hyperparameters'] = results['hyperparameters'][np.argmax(results['scores'])]
    results['best_score'] = np.max(results['scores'])
    return results


if __name__ == '__main__':
    # Create a sample dataset
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, size=100)
    dataset = Dataset(X, y)

    # Define hyperparameter grid
    hyperparameter_grid = {
        'param1': (1, 2, 3),
        'param2': (0.1, 0.2, 0.3),
        'param3': ('a', 'b', 'c')
    }

    # Define a scoring function (if needed)
    def custom_scoring_function(true_labels, predicted_labels):
        # Your custom scoring logic here
        return 0.5  # Replace this with your scoring calculation

    # Call grid_search_cv function
    results = grid_search_cv(YourModel(), dataset, hyperparameter_grid, scoring=custom_scoring_function, cv=3)

    # Print the results
    print("Grid Search Results:")
    print("Best Hyperparameters:", results['best_hyperparameters'])
    print("Best Score:", results['best_score'])
    print("All Scores:", results['scores'])
    print("All Hyperparameters:", results['hyperparameters'])