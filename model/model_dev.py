import logging
from abc import ABC, abstractmethod


import pandas as pd
from sklearn.linear_model import LinearRegression
from typing import Dict
import optuna  # Import the optuna library

# Rest of your code...

class Model(ABC):
    """
    Abstract base class for all models.
    """

    @abstractmethod
    def train(self, X_train, y_train):
        """
        Trains the model on the given data.

        Args:
            x_train: Training data
            y_train: Target data
        """
        pass

class LinearRegressionModel(Model):
    """
    LinearRegressionModel that implements the Model interface.
    """

    def train(self, X_train, y_train, **kwargs):
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            logging.info('Training complete')
            return reg
        except Exception as e:
            logging.error("error in trainig model ".format(e))
            raise e