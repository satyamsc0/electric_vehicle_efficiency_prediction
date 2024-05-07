import sys
sys.path.append('d:/Jio_Institute/electric_vehicle_efficiency_prediction')

import logging
import mlflow
import numpy as np
import pandas as pd
from model.evaluation import MSE, RMSE, R2
from sklearn.base import RegressorMixin
from typing_extensions import Annotated
from zenml import step

from typing import Tuple
from zenml.client import Client 
experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(
    model: RegressorMixin, X_test: pd.DataFrame, y_test: pd.Series
) -> Tuple[Annotated[float, "r2"], 
           Annotated[float, "rmse"],
]:
    """
    Args:
        model: RegressorMixin
        x_test: pd.DataFrame
        y_test: pd.Series
    Returns:
        r2_score: float
        rmse: float
    """
    try:
        prediction = model.predict(X_test)

        # Using the MSE class for mean squared error calculation
        mse_class = MSE()
        mse = mse_class.calculate_score(y_test, prediction)
        mlflow.log_metric("mse", mse)

        # Using the R2Score class for R2 score calculation
        r2_class = R2()
        r2 = r2_class.calculate_score(y_test, prediction)
        mlflow.log_metric("r2", r2)
        # Using the RMSE class for root mean squared error calculation
        rmse_class = RMSE()
        rmse = rmse_class.calculate_score(y_test, prediction)
        mlflow.log_metric("rmse", rmse)

        return r2, rmse
    except Exception as e:
        logging.error("error in evaluation".format(e))
        raise e
