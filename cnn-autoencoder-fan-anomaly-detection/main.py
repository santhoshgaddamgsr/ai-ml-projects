# ===============================
# CNN Autoencoder for Fan Anomaly Detection
# FINAL – Correct Threshold Logic
# ===============================

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

# ------------------------------
# GPU SETUP
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ------------------------------
# DATA PATH (portable)
# ------------------------------
DATA_PATH = os.path.join(os.getcwd(), "data")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Data folder not found.\n"
        f"Expected structure:\n"
        f"project_root/data/*.npy\n"
        f"Current path: {DATA_PATH}"
    )

print("Dataset path:", DATA_PATH)

# ------------------------------
# 1️⃣ Load Data
# ------------------------------
X_train = np.load(os.path.join(DATA_PATH, "dc2020t2l1-fan-train.npy"))
X_test  = np.load(os.path.join(DATA_PATH, "dc2020t2l1-fan-test.npy"))

print("Train shape:", X_train.shape)
print("Test shape :", X_test.shape)

# Add channel dimension
X_train = X_train[:, np.newaxis, :, :]
X_test  = X_test[:, np.newaxis, :, :]

# Normalize
X_train = X_train / np.max(X_train)
X_test  = X_test / np.max(X_test)

# Torch tensors
X_train_torch = torch.tensor(X_train, dtype=torch.float32)
X_test_torch  = torch.tensor(X_test, dtype=torch.float32)

train_loader = DataLoader(
    TensorDataset(X_train_torch, X_train_torch),
    batch_size=32,
    shuffle=True
)

# ------------------------------
# 2️⃣ CNN AUTOENCODER
# ------------------------------
class CNNAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(8, 8, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(8, 8, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),

            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        # handle odd width (313)
        if decoded.shape[3] < x.shape[3]:
            decoded = F.pad(decoded, (0, x.shape[3] - decoded.shape[3]))

        if decoded.shape[3] > x.shape[3]:
            decoded = decoded[:, :, :, :x.shape[3]]

        return decoded


model = CNNAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(model)

# ------------------------------
# 3️⃣ TRAIN
# ------------------------------
EPOCHS = 20

for epoch in range(EPOCHS):
    total_loss = 0

    for batch_x, _ in train_loader:
        batch_x = batch_x.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_x)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss/len(train_loader):.6f}")

# ------------------------------
# 4️⃣ TRAIN RECONSTRUCTION ERROR
# ------------------------------
model.eval()
with torch.no_grad():
    train_recon = model(X_train_torch.to(device)).cpu()

train_mse = torch.mean(
    (X_train_torch - train_recon) ** 2,
    dim=(1,2,3)
).numpy()

threshold = np.percentile(train_mse, 95)
print("Threshold (95th percentile):", threshold)

# ------------------------------
# 5️⃣ TEST
# ------------------------------
with torch.no_grad():
    test_recon = model(X_test_torch.to(device)).cpu()

test_mse = torch.mean(
    (X_test_torch - test_recon) ** 2,
    dim=(1,2,3)
).numpy()

y_pred = (test_mse > threshold).astype(int)

# ------------------------------
# 6️⃣ EVALUATION
# ------------------------------
df = pd.read_csv(os.path.join(DATA_PATH, "file_info.csv"))
df_test = df[(df.type == "fan") & (df.split == "test")].reset_index(drop=True)

y_true = df_test.file.str.contains("anomaly").astype(int)

acc = accuracy_score(y_true, y_pred)
f1  = f1_score(y_true, y_pred)

print(f"Accuracy: {acc*100:.2f}%")
print(f"F1-score: {f1:.2f}")

# ------------------------------
# 7️⃣ PLOT
# ------------------------------
plt.figure(figsize=(10,4))
plt.plot(test_mse, label="Reconstruction Error")
plt.axhline(threshold, linestyle="--", label="Threshold")
plt.title("CNN Autoencoder – Fan Anomaly Detection")
plt.legend()
plt.show()
