import logging
import pandas as pd
from zenml import step

class IngestData:
    """Ingests data from a CSV file."""

    def __init__(self, data_path: str):
        self.data_path = data_path

    def get_data(self):
        logging.info(f"Ingest data from {self.data_path}")
        return pd.read_csv(self.data_path)

@step
def ingest_df(data_path: str) -> pd.DataFrame:
    """
    Ingest data from the specified path and return a DataFrame.

    Args:
        data_path (str): The path to the data file.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the ingested data.
    """
    try:
        ingest_df = IngestData(data_path)
        df = ingest_df.get_data()
        return df
    except Exception as e:
        logging.error(f"Error while ingesting data: {e}")
        raise e
