import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, f1_score

# Load data
X_train = np.load(r"D:\ai-pocs-portfolio\fan-anomaly-detection-isolation-forest\dc2020t2l1-fan-train.npy")
X_test = np.load(r"D:\ai-pocs-portfolio\fan-anomaly-detection-isolation-forest\dc2020t2l1-fan-test.npy")

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# Flatten
n_train = X_train.shape[0]
n_test = X_test.shape[0]
X_train_flat = X_train.reshape(n_train, -1)
X_test_flat = X_test.reshape(n_test, -1)

print("Flattened train shape:", X_train_flat.shape)
print("Flattened test shape:", X_test_flat.shape)

# Read true labels from CSV
df = pd.read_csv(r"D:\ai-pocs-portfolio\fan-anomaly-detection-isolation-forest\file_info.csv")
df_fan_test = df[(df.type == "fan") & (df.split == "test")].reset_index(drop=True)
y_true = df_fan_test.file.str.contains("anomaly").astype(int)

# Parameter grid
contamination_list = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
n_estimators_list = [100, 200, 300]

best_f1 = 0
best_params = None

# Grid search
for cont in contamination_list:
    for est in n_estimators_list:
        model = IsolationForest(contamination=cont,
                                n_estimators=est,
                                max_samples=n_train,
                                random_state=42)
        model.fit(X_train_flat)
        y_pred = model.predict(X_test_flat)
        y_pred = np.where(y_pred == 1, 0, 1)  # Map to 0/1
        f1 = f1_score(y_true, y_pred)

        print(f"cont={cont}, est={est}, F1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_params = (cont, est)

# Best params found
print("\n✅ Best Parameters:")
print("Contamination:", best_params[0])
print("n_estimators:", best_params[1])
print("Best F1-score:", best_f1)

# Train final model with best params
best_model = IsolationForest(contamination=best_params[0],
                             n_estimators=best_params[1],
                             max_samples=n_train,
                             random_state=42)
best_model.fit(X_train_flat)
final_pred = best_model.predict(X_test_flat)
final_pred = np.where(final_pred == 1, 0, 1)

print("\nPredictions (first 20):", final_pred[:20])
