import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import json

# Load settings from settings.json
with open("settings.json", "r") as file:
    conf = json.load(file)

# Define paths and parameters from settings
DATA_DIR = conf["general"]["data_dir"]
TRAIN_FILE = conf["train"]["table_name"]
TEST_FILE = conf["inference"]["inp_table_name"]
TEST_SIZE = conf["train"]["test_size"]
RANDOM_STATE = conf["general"]["random_state"]

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# URL for the Iris dataset
IRIS_DATASET_URL = (
    "https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv"
)


# Function to download and preprocess the Iris dataset
def download_and_preprocess():
    # Download dataset
    df = pd.read_csv(IRIS_DATASET_URL)

    # Encode categorical labels
    encoder = LabelEncoder()
    df["species"] = encoder.fit_transform(df["species"])

    # Split dataset into features and target
    X = df.drop("species", axis=1)
    y = df["species"]

    # Split dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Scale features - fit on training data and transform both training and test data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save preprocessed datasets
    train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    train_df["species"] = y_train.reset_index(drop=True)
    train_df.to_csv(os.path.join(DATA_DIR, TRAIN_FILE), index=False)

    test_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    test_df["species"] = y_test.reset_index(drop=True)
    test_df.to_csv(os.path.join(DATA_DIR, TEST_FILE), index=False)

    print("Data downloaded and preprocessed successfully.")


# Main execution
if __name__ == "__main__":
    download_and_preprocess()
