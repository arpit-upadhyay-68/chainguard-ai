import joblib
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import os

# 1. Automatically find the exact absolute path to your artifacts folder
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(CURRENT_DIR, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "disruption_model.joblib")

# 2. Train a dummy model (replace with your real dataset later)
print("Training model...")
X = np.random.rand(100, 4) 
y = np.random.randint(0, 2, 100)
model = RandomForestClassifier().fit(X, y)

# 3. Save the model
joblib.dump(model, MODEL_PATH)
print(f"✅ Model successfully saved at: {MODEL_PATH}")