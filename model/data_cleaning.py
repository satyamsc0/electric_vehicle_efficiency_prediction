import logging
import numpy as np
import pandas as pd
from typing import Union
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split


class DataStrategy(ABC):
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass

class DataPreProcessStrategy(DataStrategy):
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:
            data.drop(['Name', 'Subtitle', 'Drive',  'NumberofSeats'], axis=1, inplace=True)

            # Clean the 'PriceinGermany' and 'PriceinUK' columns by removing '€' and '£' signs
            data['PriceinGermany'] = data['PriceinGermany'].str.replace('[€,]', '', regex=True).astype(float)
            data['PriceinUK'] = data['PriceinUK'].str.replace('[£,]', '', regex=True).astype(float)

            # Remove units from 'Range', 'Efficiency', and 'FastChargeSpeed' columns
            data['Range'] = data['Range'].str.replace(' km', '').astype(float)
            data['Efficiency'] = data['Efficiency'].str.replace(' Wh/km', '').astype(float)

            # Handle 'FastChargeSpeed' column
            data['FastChargeSpeed'] = data['FastChargeSpeed'].str.replace(' km/h', '')
            # Replace non-numeric values ('-') with NaN
            data['FastChargeSpeed'] = data['FastChargeSpeed'].replace('-', np.nan)
            # Convert the column to float
            data['FastChargeSpeed'] = data['FastChargeSpeed'].astype(float)

            # Remove 'sec' from the 'Acceleration' column
            data['Acceleration'] = data['Acceleration'].str.replace(' sec', '').astype(float)

            # Remove ' km/h' from the 'TopSpeed' column
            data['TopSpeed'] = data['TopSpeed'].str.replace(' km/h', '').astype(float)

            # Fill N/A values with the mean of each column
            data.fillna(data.mean(), inplace=True)

            # Print the cleaned DataFrame
            print(data)
            return data
        except Exception as e:
            logging.error("error in data_cleaning".format(e))
            raise e


class DataDivideStrategy(DataStrategy):
    """
    Data dividing strategy which divides the data into train and test data.
    """

    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Divides the data into train and test data.
        """
        try:
            # Assuming "Efficiency" is your target variable
            X = data.drop("Efficiency", axis=1)
            y = data["Efficiency"]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error("Error in Divides the data into train and test data.".format(e))
            raise e

class DataCleaning:
    """
    Data cleaning class which preprocesses the data and divides it into train and test data.
    """

    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        """Initializes the DataCleaning class with a specific strategy."""
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """Handle data based on the provided strategy"""
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error("error in handeling data".format(e))
            raise e