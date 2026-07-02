import logging
import pandas as pd
import numpy as np
from sklearn.base import ClassifierMixin
from src.model_evaluation import F1ScoreEvaluate, ConfusionMatrixEvaluate
from zenml import step
from typing import Tuple
from typing_extensions import Annotated


@step
def model_evaluation(
    model: ClassifierMixin,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[
    Annotated[float, "f1_score_result"],
    Annotated[np.ndarray, "con_matrix_result"]
]:
    """Evaluate the trained model's performance on test data.

    Args:
        model: A trained Scikit-learn classifier model object.
        X_test: Testing features DataFrame.
        y_test: Testing target labels Series.

    Returns:
        Tuple containing the F1 score (float) and the confusion matrix (ndarray).
    """

    try:
        logging.info("Generating predictions on the test dataset.")
        prediction = model.predict(X_test)

        f1 = F1ScoreEvaluate()
        f1_score_result = f1.evaluate(y_test, prediction)
        
        cm = ConfusionMatrixEvaluate()
        con_matrix_result = cm.evaluate(y_test, prediction)
        logging.info("Pipeline step model evaluation finished successfully.")

        return f1_score_result, con_matrix_result
    
    except Exception as e:
        logging.error(f"Error during model evaluation step execution {e}.")
        raise e
    