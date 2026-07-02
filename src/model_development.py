import logging

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier


class ModelDevelopment(ABC):
    """A abstract class for model development and optimization"""

    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs):
        """Train the model.
        
        Args: 
            X_train(pd.DataFrame): Contains the training features.
            y_train(pd.Series): Contains the target variable.
        Returns: 
            Classifier object.
        """ 
        pass



class RandomForestModel(ModelDevelopment):
    """RandomForestClassifier Model wrapper."""

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs):
        """Train the Random Forest model.
        
        Args: 
            X_train(pd.DataFrame): Contains the training features.
            y_train(pd.Series): Contains the target variable.
        Returns: 
            RandomForestClassifier: The trained model instance.
        """ 
        
        try:
            logging.info("Model training started.")
            rf = RandomForestClassifier(**kwargs)
            rf.fit(X_train, y_train)
            logging.info("Model training completed.")

            return rf
        except Exception as e:
            logging.error(f"Error while training model {e}.")
            raise e 