# training.py
import torch
from models.baseline import BaselineRegressor
from models.lipschitz_model import LipschitzRegressor
from Data.DataLoader import load_data
import torch.nn as nn
import torch.optim as optim
import torch
from config import BATCH_SIZE, LEARNING_RATE, EPOCHS

# --- Load Data ---
X_train, X_test, y_train, y_test, train_loader = load_data(batch_size=BATCH_SIZE)

# --- Baseline Regressor ---
baseline_model = BaselineRegressor(X_train.shape[1])
optimizer_baseline = optim.Adam(baseline_model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

for epoch in range(EPOCHS):
    baseline_model.train()
    total_loss = 0
    for xb, yb in train_loader:
        preds = baseline_model(xb)
        loss = loss_fn(preds, yb)
        optimizer_baseline.zero_grad()
        loss.backward()
        optimizer_baseline.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

torch.save(baseline_model.state_dict(), "baseline_model.pth")

# ---  Lipschitz  Regressor ---
model_Lipschitz = LipschitzRegressor(X_train.shape[1])
optimizer_lipschitz = optim.Adam(model_Lipschitz.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    model_Lipschitz.train()
    total_loss = 0
    for xb, yb in train_loader:
        preds = model_Lipschitz(xb)
        loss = loss_fn(preds, yb)
        optimizer_lipschitz.zero_grad()
        loss.backward()
        optimizer_lipschitz.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

torch.save(model_Lipschitz.state_dict(), "lipschitz_model.pth")