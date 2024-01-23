import os
import sys
import json
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# Define your model structure (must match the structure used in train.py)
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(4, 10)  # Assuming input size is 4, hidden size is 10
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 3)  # Assuming output size (num_classes) is 3

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# Adds the root directory to system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

# Load settings from settings.json
CONF_FILE = os.getenv("CONF_PATH", "settings.json")
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

DATA_DIR = conf["general"]["data_dir"]
MODEL_DIR = conf["general"]["models_dir"]
INFERENCE_FILE = conf["inference"]["inp_table_name"]
RESULTS_DIR = conf["general"]["results_dir"]


def load_data(path):
    df = pd.read_csv(path)
    X = df.values
    X_tensor = torch.tensor(X, dtype=torch.float32)
    return X_tensor


def inference(model, data_loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for features in data_loader:
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.tolist())
    return predictions


def main():
    # Load model
    model = SimpleNN()
    model_path = os.path.join(MODEL_DIR, "pytorch_model.pth")
    model.load_state_dict(torch.load(model_path))

    # Prepare data
    inference_data_path = os.path.join(DATA_DIR, INFERENCE_FILE)
    inference_data = load_data(inference_data_path)
    inference_loader = DataLoader(TensorDataset(inference_data), batch_size=32)

    # Run inference
    results = inference(model, inference_loader)

    # Save results
    results_path = os.path.join(RESULTS_DIR, "inference_results.csv")
    pd.DataFrame(results, columns=["Predictions"]).to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
