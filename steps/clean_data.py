import logging
import pandas as pd
from zenml import step

@step
def clean_df(df: pd.DataFrame) -> None:
    """
    A step to clean the dataframe.
    Args:
        df: The input dataframe.
    Returns:
        pd.DataFrame: The cleaned dataframe.
    """
    pass 