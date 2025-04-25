# evaluation.py
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from Data.DataLoader import load_data
from models.baseline import BaselineRegressor
from models.lipschitz_model import LipschitzRegressor


X_train, X_test, y_train, y_test, train_loader = load_data(batch_size=32)


baseline_model = BaselineRegressor(X_train.shape[1])
baseline_model.load_state_dict(torch.load("baseline_model.pth"))

model_Lipschitz = LipschitzRegressor(X_train.shape[1])
model_Lipschitz.load_state_dict(torch.load("lipschitz_model.pth"))

baseline_model.eval()
with torch.no_grad():
    y_pred_baseline = baseline_model(X_test)

model_Lipschitz.eval()
with torch.no_grad():
    y_pred_lipschitz = model_Lipschitz(X_test)

mse_baseline = mean_squared_error(y_test, y_pred_baseline)
r2_baseline = r2_score(y_test, y_pred_baseline)

mse_lipschitz = mean_squared_error(y_test, y_pred_lipschitz)
r2_lipschitz = r2_score(y_test, y_pred_lipschitz)

print(f"Baseline Model - MSE: {mse_baseline:.4f}, R²: {r2_baseline:.4f}")
print(f"Lipschitz Model - MSE: {mse_lipschitz:.4f}, R²: {r2_lipschitz:.4f}")

y_test_np = y_test.numpy()
y_pred_baseline_np = y_pred_baseline.numpy()
y_pred_lipschitz_np = y_pred_lipschitz.numpy()

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_test_np, y_pred_baseline_np, c='blue', alpha=0.5)
plt.plot([min(y_test_np), max(y_test_np)], [min(y_test_np), max(y_test_np)], 'k--', lw=2)
plt.title("Baseline Model: Predictions vs True Values")
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(y_test_np, y_pred_lipschitz_np, c='red', alpha=0.5)
plt.plot([min(y_test_np), max(y_test_np)], [min(y_test_np), max(y_test_np)], 'k--', lw=2)
plt.title("Lipschitz Model: Predictions vs True Values")
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.grid(True)

plt.tight_layout()
plt.show()
