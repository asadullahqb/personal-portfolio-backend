import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import mlflow
from app.config import settings

# Generate dummy dataset
X, y = np.random.rand(200, 4), np.random.randint(0, 2, 200)
df = pd.DataFrame(X, columns=["feat1", "feat2", "feat3", "feat4"])
df["target"] = y

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(df.drop("target", axis=1),
                                                    df["target"], test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"Model A accuracy: {acc}")

# MLflow tracking
mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
mlflow.set_experiment("model_a")
with mlflow.start_run():
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")

# Save locally
joblib.dump(model, settings.model_a_path)
