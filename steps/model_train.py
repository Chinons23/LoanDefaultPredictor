import logging
import pandas as pd
from zenml import step

@step
def train_model(df: pd.DataFrame) -> None:
   """
   Train a model on the given DataFrame.
   Args:
       df: A pandas DataFrame containing the training data.
   Returns:
       None
   """
   pass 