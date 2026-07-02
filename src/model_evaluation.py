import logging

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Union
from sklearn.metrics import f1_score, confusion_matrix

# To both NumPy and Pandas inputs
ArrayLike = Union[np.ndarray, pd.Series]

class EvaluateModel(ABC):
    """A abstract class for evaluating model."""

    @abstractmethod
    def evaluate(self, y_true: ArrayLike, y_pred: ArrayLike) -> Union[float, np.ndarray]:
        """Evaluate the model predictions against true labels.
        
        Args: 
            y_true: True target values.
            y_pred: Predicted target values from the model.
        Returns: 
            The calculated evaluation metric result.
        """
        pass


class F1ScoreEvaluate(EvaluateModel):
    """Calculates the F1 score for classification models."""

    def evaluate(self, y_true: ArrayLike, y_pred: ArrayLike, **kwargs) -> float:
        """Evaluate the model using the F1 score.

        Args: 
            y_true: True target values.
            y_pred: Predicted target values from the model.
            **kwargs: Additional arguments passed to sklearn.metrics.f1_score.
        Returns: 
            float: The computed F1 score.
        """

        try:
            logging.info("Model Evaluation using F1 score Started.")
            kwargs.setdefault('average', 'binary')
            f1_result = f1_score(y_true, y_pred, **kwargs)
            logging.info(f"Model Evaluated. F1 score: {f1_result:.3f}.")
            return f1_result
        
        except Exception as e:
            logging.error(f"Error while evaluating F1 score: {e}.")
            raise e


class ConfusionMatrixEvaluate(EvaluateModel):
    """Generates the confusion matrix for classification models."""

    def evaluate(self, y_true: ArrayLike, y_pred: ArrayLike, **kwargs) -> np.ndarray:
        """Evaluate the model using a confusion matrix.

        Args: 
            y_true: Ground truth target values.
            y_pred: Predicted target values from the model.
            **kwargs: Additional arguments passed to sklearn.metrics.confusion_matrix.
        Returns: 
            np.ndarray: Confusion matrix array.
        """

        try:
            logging.info("Model evaluation using Confusion Matrix started.")
            cm_result = confusion_matrix(y_true, y_pred, **kwargs)
            logging.info(f"Model evaluation completed. Confusion Matrix:\n{cm_result}.")
            return cm_result
        
        except Exception as e:
            logging.error(f"Error while evaluating Confusion Matrix: {e}.")
            raise e
