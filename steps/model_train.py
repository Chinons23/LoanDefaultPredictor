import logging
import pandas as pd
from zenml import step
from src.model_development import RandomForestModel
from sklearn.base import ClassifierMixin
from .config import MyParameter

@step
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: MyParameter = MyParameter()
) -> ClassifierMixin:
    """Train a model on the given features and targets.

    Args:
        X_train: Features for training.
        X_test: Features for testing (currently unused in this step).
        y_train: Target labels for training.
        y_test: Target labels for testing (currently unused in this step).
        config: Configuration parameters containing the model name.

    Returns:
        ClassifierMixin: The trained Scikit-learn classifier model object.
    """
    try:
        # Check model name from ZenML configuration parameter class
        if config.model_name == "RandomForestClassifier":
            logging.info("Training Model Started.")
            model = RandomForestModel()
            rf_model = model.train(X_train, y_train)
            logging.info("Training completed.")
            return rf_model
        
        # Explicitly raise an error if an unsupported model is passed
        raise ValueError(f"Model '{config.model_name}' is not supported.")
        
    except Exception as e:
        logging.error(f"Error while selecting model {e}.")
        raise e
