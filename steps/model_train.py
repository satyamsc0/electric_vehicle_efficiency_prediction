import sys
sys.path.append('d:/Jio_Institute/electric_vehicle_efficiency_prediction')

import logging
import pandas as pd
from model.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from zenml import step
import mlflow 
from zenml.client import Client 
experiment_tracker = Client().active_stack.experiment_tracker



from .config import ModelNameConfig

@step                   #(experiment_tracker=experiment_tracker.name)
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: ModelNameConfig,
) -> RegressorMixin:
    """
    Train a regression model based on the specified configuration.

    Args:
        X_train (pd.DataFrame): Training data features.
        X_test (pd.DataFrame): Testing data features.
        y_train (pd.Series): Training data target.
        y_test (pd.Series): Testing data target.
        config (ModelNameConfig): Model configuration.

    Returns:
        RegressorMixin: Trained regression model.
    """
    try:
        model = None
        if config.model_name == "linear_regression":
            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
            train_model = model.train(X_train, y_train)
            return train_model
        else:
            raise ValueError("Model name not supported")
    except Exception as e:
        logging.error("error in train model ".format(e))
        raise e
