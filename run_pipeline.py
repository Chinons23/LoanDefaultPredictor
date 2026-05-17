from pipelines.training_pipeline import train_pipeline


if __name__ == "__main__":
    # Run the training pipeline
    train_pipeline(data_path="data\loan_data.csv")