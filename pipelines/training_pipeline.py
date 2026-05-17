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
    clean_df(df)
    train_model(df)
    model_evaluation(df)