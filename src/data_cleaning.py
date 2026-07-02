import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Tuple
from sklearn.model_selection import train_test_split



class DataStrategy(ABC):
    """Data Strategy class for defining how to handle data."""

    @abstractmethod
    def treat_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        pass


class DataPreprocessing(DataStrategy):
    """
    Prepcoessing class that clean and transform the dataset.
    """

    def treat_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Load the raw data clean and tranform it for model training.

        Args:
            data: This contains the dataframe to be process

        Returns:
            data: A clean data ready
        """
        
        try:
            logging.info("Dropping irrelevant columns.")
            data = data.drop(['purpose'], axis=1, errors='ignore')
            logging.info(f"Columns dropped. Remaining columns: {list(data.columns)}")
            return data
        except Exception as e:
            logging.error(f"Error while processing data {e}.")
            raise e
        

class DataSplitStrategy(DataStrategy):
    """
    Strategy class to split data
    """

    def treat_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split the give data into training and testing set.

        Args:
            X: (pd.DataFrame): The input dataframe to be split.
            y: (pd.Series): The target variable

        Returns:
            X_train, X_test, y_train, y_test: Split data
        """

        try:
            X = data.drop(columns='not.fully.paid')
            y = data['not.fully.paid']

            logging.info("Splitting data into training and testing set.")
            # Split the dataset into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                                test_size=0.2, 
                                                                stratify=y,
                                                                random_state=42)
            logging.info(
                f"Successfully splitted into train data {X_train.shape, y_train.shape},"
                f"and test data {X_test.shape, y_test.shape}."
                )
            
            return X_train, X_test, y_train, y_test
        
        except Exception as e:
            logging.error(f"Error while splitting data {e}.")
            raise e
        
class ProcessStrategy:
    """
    Context class to manage and execute data processing strategies.

    Args:
        data (pd.DataFrame): The input DataFrame to process.
        strategy (DataStrategy): The specific strategy implementation to apply.
    """

    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy

    def treat_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Executes the assigned data strategy on the input data
        """
        try:
            logging.info(f"Starting data processing using {self.strategy.__class__.__name__}.")
            return self.strategy.treat_data(self.data)
        
        except Exception as e:
            logging.error("Error while executing data strategy.", exc_info=True)
            raise e