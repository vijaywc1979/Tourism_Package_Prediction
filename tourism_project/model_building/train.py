
import os
import joblib
import pandas as pd
from datasets import load_dataset
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from huggingface_hub import HfApi

# ========== CONFIG ==========
TRAIN_REPO = "vijaywc1979/Tourism-Package-Train"
TEST_REPO = "vijaywc1979/Tourism-Package-Test"
MODEL_REPO = "vijaywc1979/Tourism-Package-DecisionTree"
MODEL_FILE = "decision_tree_model.pkl"

# ========== LOAD DATA FROM HF ==========
train_df = load_dataset(TRAIN_REPO)["train"].to_pandas()
test_df = load_dataset(TEST_REPO)["train"].to_pandas()

X_train = train_df.drop(columns=["ProdTaken"])
y_train = train_df["ProdTaken"]

X_test = test_df.drop(columns=["ProdTaken"])
y_test = test_df["ProdTaken"]

print("Train and test datasets loaded from Hugging Face.")

# ========== MODEL DEFINITION ==========
dt = DecisionTreeClassifier(random_state=42)

# ========== PARAMETER TUNING (LIGHTWEIGHT) ==========
param_grid = {
    "max_depth": [3, 5, 7],
    "min_samples_split": [2, 5]
}

grid = GridSearchCV(
    estimator=dt,
    param_grid=param_grid,
    cv=3,
    scoring="accuracy"
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_
best_params = grid.best_params_

print("Best Parameters:", best_params)

# ========== EVALUATION ==========
preds = best_model.predict(X_test)
accuracy = accuracy_score(y_test, preds)

print("Test Accuracy:", accuracy)
print(classification_report(y_test, preds))

# ========== SAVE MODEL ==========
joblib.dump(best_model, MODEL_FILE)
print("Model saved locally.")

# ========== REGISTER MODEL TO HF ==========
api = HfApi(token=os.getenv("HF_TOKEN"))

api.upload_file(
    path_or_fileobj=MODEL_FILE,
    path_in_repo=MODEL_FILE,
    repo_id=MODEL_REPO,
    repo_type="model"
)

print("Model successfully registered on Hugging Face Model Hub.")
