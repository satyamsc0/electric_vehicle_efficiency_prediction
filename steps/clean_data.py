import sys
sys.path.append('d:/Jio_Institute/electric_vehicle_efficiency_prediction')

import logging
from typing import Tuple

import pandas as pd
from model.data_cleaning import DataCleaning, DataDivideStrategy, DataPreProcessStrategy
from zenml import step
from typing_extensions import Annotated

@step
def clean_df(data: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, 'X_train'],
    Annotated[pd.DataFrame, 'X_test'],
    Annotated[pd.Series, 'y_train'],
    Annotated[pd.Series, 'y_test'],
]:
    """
    Data cleaning class which preprocesses the data and divides it into train and test data.

    Args:
        data: pd.DataFrame
    """
    try:
        preprocess_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(data, preprocess_strategy)
        preprocessed_data = data_cleaning.handle_data()

        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(preprocessed_data, divide_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        logging.info(f"Data Cleaning Complete")
        return X_train, X_test, y_train, y_test 
    except Exception as e: 
        logging.error(e)
        raise e
