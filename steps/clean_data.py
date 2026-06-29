import logging
import pandas as pd
from zenml import step
from typing_extensions import Annotated
from typing import Tuple
from src.data_cleaning import ProcessStrategy, DataPreprocessing, DataSplitStrategy

@step
def clean_df(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"]
]:
    """A step to clean and split the dataframe using a strategy pattern.

    Args:
        df: The input dataframe.

    Returns:
        X_train: Training data
        X_test: Testing data
        y_train: Training label
        y_test: Testing label
    """
    try:
        logging.info("Executing Data Preprocessing strategy.")
        preprocess_strategy = DataPreprocessing()
        pipeline = ProcessStrategy(df, preprocess_strategy)
        preprocess_data = pipeline.treat_data()
        logging.info("Preprocessing strategy completed.")
        
        logging.info("Executing Data Split strategy.")
        split_strategy = DataSplitStrategy()
        pipeline = ProcessStrategy(preprocess_data, split_strategy)
        X_train, X_test, y_train, y_test = pipeline.treat_data()
        logging.info("Split strategy completed.")

        return X_train, X_test, y_train, y_test
    
    except Exception as e:
        logging.error(f"Error while executing strategy {e}")
        raise e