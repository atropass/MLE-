import os
import json
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging
from sklearn.metrics import classification_report


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(4, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 3)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


CONF_FILE = "settings.json"
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

DATA_DIR = conf["general"]["data_dir"]
MODEL_DIR = conf["general"]["models_dir"]
INFERENCE_FILE = conf["inference"]["inp_table_name"]
RESULTS_DIR = conf["general"]["results_dir"]


def load_data(path):
    df = pd.read_csv(path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    return X_tensor, y_tensor


def inference(model, data_loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for features, _ in data_loader:
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.tolist())
    return predictions


def calculate_accuracy(actual, predicted):
    correct = sum(a == p for a, p in zip(actual, predicted))
    return correct / len(actual)


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    model = SimpleNN()
    model_path = os.path.join(MODEL_DIR, "pytorch_model.pth")
    model.load_state_dict(torch.load(model_path))

    inference_data_path = os.path.join(DATA_DIR, INFERENCE_FILE)
    X, y = load_data(inference_data_path)
    inference_dataset = TensorDataset(X, y)
    inference_loader = DataLoader(inference_dataset, batch_size=32, shuffle=False)

    predictions = inference(model, inference_loader)

    report = classification_report(
        y.numpy(), predictions, target_names=["Class 0", "Class 1", "Class 2"]
    )
    logging.info(f"\nClassification Report:\n{report}")

    results_df = pd.DataFrame({"Actual": y.numpy(), "Predicted": predictions})
    results_path = os.path.join(RESULTS_DIR, "inference_results.csv")
    results_df.to_csv(results_path, index=False)
    logging.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
