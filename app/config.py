import os
import xgboost as xgb

# Get the model path from an environment variable or default to the local JSON file
MODEL_PATH = os.getenv("MODEL_PATH", "/model/xgboost_model.json")

# load the model
model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)
