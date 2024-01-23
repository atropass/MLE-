import argparse
import os
import sys
import json
import logging
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Comment this line if you have problems with MLFlow installation
import mlflow

mlflow.autolog()

# Adds the root directory to system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

# Change to CONF_FILE = "settings.json" if you have problems with env variables
CONF_FILE = os.getenv("CONF_PATH")

from utils import get_project_dir, configure_logging

with open(CONF_FILE, "r") as file:
    conf = json.load(file)

DATA_DIR = get_project_dir(conf["general"]["data_dir"])
MODEL_DIR = get_project_dir(conf["general"]["models_dir"])
TRAIN_PATH = os.path.join(DATA_DIR, conf["train"]["table_name"])

input_size = 4
hidden_size = 10
num_classes = 3
learning_rate = 0.001
num_epochs = 50


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


def prepare_data(path, test_size=0.2):
    df = pd.read_csv(path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    return (X_train_tensor, y_train_tensor), (X_test_tensor, y_test_tensor)


def train(model, criterion, optimizer, data_loader):
    model.train()
    for epoch in range(num_epochs):
        for features, labels in data_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in data_loader:
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def main():
    (X_train, y_train), (X_test, y_test) = prepare_data(TRAIN_PATH)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = SimpleNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train(model, criterion, optimizer, train_loader)

    accuracy = evaluate(model, test_loader)
    print(f"Test Accuracy: {accuracy:.4f}")

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    model_path = os.path.join(MODEL_DIR, "pytorch_model.pth")
    torch.save(model.state_dict(), model_path)

    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
