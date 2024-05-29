import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
import numpy as np


def train_model(X_train, y_train, model_type, search_space: dict):
    """
    Train a machine learning model using the specified model type and search space for hyperparameters.

    Args:
    X_train (DataFrame): Training features.
    y_train (Series): Training labels.
    model_type (str): Type of model to train.
    search_space (dict): Hyperparameter search space.

    Returns:
    RandomizedSearchCV: Trained model.
    """
    if model_type == "random_forest":
        model = RandomForestClassifier()
    elif model_type == "logistic_regression":
        model = LogisticRegression()
    elif model_type == "xgboost":
        model = XGBClassifier()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Set the desired value for zero_division
    zero_division_value = np.nan  # You can choose another value if you prefer

    # Create a custom scorer with the specified zero_division
    f1_scorer = make_scorer(f1_score, zero_division=zero_division_value)

    # Use the custom scorer when setting up RandomizedSearchCV
    randomized_search = RandomizedSearchCV(model, search_space, scoring=f1_scorer, n_jobs=-1, cv=5, verbose=10, random_state=42)
    print('Starting training...')

    randomized_search.fit(X_train, y_train)

    # Best combination of hyperparameters
    print("Best combination of hyperparameters:", randomized_search.best_params_)
    return randomized_search


def test_model(randomized_search, X_test, y_test):
    """
    Test a machine learning model and evaluate its performance.

    Args:
    randomized_search (RandomizedSearchCV): Trained model.
    X_test (DataFrame): Test features.
    y_test (Series): Test labels.

    Returns:
    DataFrame: Evaluation metrics.
    estimator: Best estimator.
    """
    # Best model
    best_model = randomized_search.best_estimator_

    from joblib import dump

    # Save the best model
    dump(randomized_search.best_estimator_, 'best_model.joblib')

    # To load the saved model
    # best_model = load('best_model.joblib')

    y_predict = best_model.predict(X_test)

    print(y_predict)
    result = {}
    #result = {'Model Name': 'Random Forest'}
    print("Random Forest Evaluation (CN vs AD) - gridsearch 5 folds")
    print("------------------------------------------------------")
    print('Score (Accuracy): ')
    print(accuracy_score(y_test, y_predict))
    result['accuracy'] = accuracy_score(y_test, y_predict)
    print("------------------------------------------------------")
    print('Precision Score (tp / (tp + fp)):')
    print(precision_score(y_test, y_predict))
    result['precision_score'] = precision_score(y_test, y_predict)
    print("------------------------------------------------------")
    print('Recall Score (tp / (tp + fn)):')
    print(recall_score(y_test, y_predict))
    result['recall_score'] = recall_score(y_test, y_predict)
    print("------------------------------------------------------")
    print('F1 Score (F1 = 2 * (precision * recall) / (precision + recall) ):')
    print(f1_score(y_test, y_predict))
    result['f1_score'] = f1_score(y_test, y_predict)
    
    return result, randomized_search.best_estimator_
