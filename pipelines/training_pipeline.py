from zenml import pipeline
from steps.ingest_data import ingest_df
from steps.clean_data import clean_df
from steps.model_train import train_model
from steps.evaluate_model import model_evaluation

@pipeline
def train_pipeline(data_path: str):
    """
    A pipeline to train a model.
    Args:
        data_path: The path to the data file is stored.
    """
    df = ingest_df(data_path)
    X_train, X_test, y_train, y_test = clean_df(df)
    model = train_model(X_train, X_test, y_train, y_test)
    f1_score_result, con_matrix_result = model_evaluation(model, X_test, y_test)