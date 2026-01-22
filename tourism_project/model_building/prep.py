
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import load_dataset, Dataset
from huggingface_hub import login

# ========== CONFIG ==========
HF_DATASET_REPO = "vijaywc1979/Tourism-Package-Prediction"
TARGET_COL = "ProdTaken"
OUTPUT_DIR = "tourism_project/data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== LOAD DATA FROM HF ==========
dataset = load_dataset(HF_DATASET_REPO)
df = dataset["train"].to_pandas()
print("Dataset loaded from Hugging Face.")

# ========== DATA CLEANING ==========
# Drop unique identifier
df.drop(columns=["CustomerID"], inplace=True)

# Handle missing values
for col in df.select_dtypes(include="number").columns:
    df[col].fillna(df[col].median(), inplace=True)

for col in df.select_dtypes(include="object").columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Encode categorical columns
label_encoder = LabelEncoder()
cat_cols = df.select_dtypes(include="object").columns

for col in cat_cols:
    df[col] = label_encoder.fit_transform(df[col])

print("Data cleaning and encoding completed.")

# ========== TRAIN-TEST SPLIT ==========
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

# Save locally
train_path = f"{OUTPUT_DIR}/train.csv"
test_path = f"{OUTPUT_DIR}/test.csv"

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

print("Train and test datasets saved locally.")

# ========== UPLOAD BACK TO HF ==========
Dataset.from_pandas(train_df).push_to_hub(
    "vijaywc1979/Tourism-Package-Train"
)

Dataset.from_pandas(test_df).push_to_hub(
    "vijaywc1979/Tourism-Package-Test"
)

print("Train and test datasets uploaded to Hugging Face.")
