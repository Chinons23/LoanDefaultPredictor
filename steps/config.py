from zenml import step 
from pydantic import BaseModel

class MyParameter(BaseModel):
    """Model Config"""
    model_name: str = "RandomForestClassifier"