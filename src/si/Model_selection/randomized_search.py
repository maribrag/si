import sys
sys.path.append("C:\\Users\\maria\\Documents\\GitHub\\si")

import numpy as np
from src.si.Model_selection.cross_validation import k_fold_cross_validation


def randomized_search_cv(model, dataset, hyperparameter_grid, scoring, cv=5, n_iter=10):
    """
    Performs randomized search cross-validation on a model.

    Parameters
    ----------
    model
        The model to cross validate.
    dataset
        The dataset to cross validate on.
    hyperparameter_grid : dict
        Dictionary with the hyperparameter name and search values.
    scoring : callable
        The scoring function to use.
    cv : int, optional
        The number of cross-validation folds, default is 5.
    n_iter : int, optional
        The number of hyperparameter random combinations to test, default is 10.

    Returns
    -------
    results : dict
        The results of the randomized search cross-validation including scores,
        hyperparameters, best hyperparameters, and best score.
    """
    # Validate the hyperparameter grid
    for parameter in hyperparameter_grid:
        if not hasattr(model, parameter):
            raise AttributeError(f"Model {model} doen't have parameter {parameter}.")

    results = {'hyperparameters': [], 'scores': []}

    for i in range(n_iter):
        # Get random hyperparameter combination
        random_combination = {param: np.random.choice(values) for param, values in hyperparameter_grid.items()}

        # Set model hyperparameters with the current combination
        for parameter, value in random_combination.items():
            setattr(model, parameter, value)

        # Cross validate the model
        scores = k_fold_cross_validation(model, dataset, scoring, cv)

        # Save mean of scores and respective hyperparameters
        results['scores'].append(np.mean(scores))
        results['hyperparameters'].append(random_combination)

    # Find best score and respective hyperparameters
    best_idx = np.argmax(results['scores'])
    best_hyperparameters = results['hyperparameters'][best_idx]
    best_score = results['scores'][best_idx]

    return {
        'hyperparameters': results['hyperparameters'],
        'scores': results['scores'],
        'best_hyperparameters': best_hyperparameters,
        'best_score': best_score
    }
    
