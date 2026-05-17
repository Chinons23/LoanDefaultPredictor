import logging
import pandas as pd
import numpy as np
from zenml import step

class IngestData:
    """
    Class to ingest data from a data folder path.
     Args:
        data_path (str): The path to the data file is stored.
    """
    def __init__(self, data_path: str):
        self.data_path = data_path


    def get_data(self):
        """
        Ingest data from the specified path and return the loaded data.
         Returns:
            pd.read_csv: Loading the ingested data as a pandas DataFrame.
        """
        logging.info("Ingesting data from {}".format(self.data_path))
        try:
            data = pd.read_csv(self.data_path)
            logging.info("Data ingestion successful")
            return data
        except Exception as e:
            logging.error("Error ingesting data: {}".format(e))
            raise e
        
@step
def ingest_df(data_path: str) -> pd.DataFrame:
    """
    Step function to ingest data from a data path.
     Args:
        data_path (str): The path to the data file is stored.
     Returns:
        pd.DataFrame: The ingested data as a pandas DataFrame.
    """
    try:
        ingest_data = IngestData(data_path)
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.error("Error in ingest_data step: {}".format(e))
        raise e