# 🌀 CNN Autoencoder for Fan Anomaly Detection

This project implements a **CNN-based Autoencoder** for **unsupervised fan sound anomaly detection** using mel-spectrogram features.

The model is trained **only on normal fan sounds** and identifies anomalous fan behavior using **reconstruction error**.

---

## 📌 Problem Statement

At the end of a production line, products must be classified as **Normal** or **Defective** before shipping.

In industrial systems, abnormal fan noise often indicates early mechanical faults.  
However, labeled defect data is limited or unavailable.

This project solves this problem using **unsupervised anomaly detection**, where the model learns only normal operating patterns and flags deviations as anomalies.

---

## 📂 Dataset & References

This project is based on the **DCASE 2020 Task 2 — Unsupervised Detection of Anomalous Sounds for Machine Condition Monitoring**.

Preprocessed fan sound features are used to focus on modeling and anomaly-detection logic rather than raw audio processing.

---

### 🔗 Reference Notebook (Preprocessed EDA)

DCASE 2020 Task 2 — Preprocessed Exploratory Data Analysis  
https://www.kaggle.com/code/muhammadmahtab/dcase-2020-task-2-preprocessed-eda/notebook

This notebook provides:
- mel-spectrogram feature extraction
- dataset exploration
- preprocessed `.npy` feature files

---

### 🔗 Dataset Files (Input)

From the following Kaggle input directory, download **only the fan-related files**:

https://www.kaggle.com/code/muhammadmahtab/dcase-2020-task-2-preprocessed-eda/input

Required files:

- `dc2020t2l1-fan-train.npy`
- `dc2020t2l1-fan-test.npy`

Each sample represents a fan sound converted into a mel-spectrogram:

(64 mel bins × 313 time frames)


---

### 🔗 Label & Metadata File (Output)

The metadata file used for evaluation can be downloaded from:

https://www.kaggle.com/code/muhammadmahtab/dcase-2020-task-2-preprocessed-eda/output

Required file:

- `file_info.csv`

This file contains:
- machine type
- train/test split
- file names
- anomaly indicators (derived from file naming)

---

## 📁 Dataset Placement

Place the downloaded files inside the `data/` folder:
```
data/
├── dc2020t2l1-fan-train.npy
├── dc2020t2l1-fan-test.npy
└── file_info.csv
```

⚠️ Dataset files are **not included** in this repository due to size and licensing restrictions.

---

## 🧠 Approach

1. Train a **CNN Autoencoder** using only normal fan sounds  
2. Learn compact representations at the bottleneck layer  
3. Reconstruct input spectrograms  
4. Compute reconstruction error using Mean Squared Error (MSE)  
5. Detect anomalies when reconstruction error exceeds a learned threshold  

---

## 🏗 Model Architecture
```
Input Spectrogram (64 × 313)
↓
Conv2D + ReLU + MaxPooling
↓
Conv2D + ReLU + MaxPooling
↓
Conv2D + ReLU + MaxPooling
↓
Bottleneck Representation
↓
Upsampling + Conv2D
↓
Upsampling + Conv2D
↓
Upsampling + Conv2D
↓
Reconstructed Spectrogram
```

- CNN layers capture local time–frequency patterns  
- Autoencoder learns only normal fan behavior  
- Abnormal sounds reconstruct poorly  

---

## 🚨 Anomaly Detection Logic

Reconstruction error is calculated as:

MSE = (Original − Reconstructed)²


### Threshold Selection

Threshold is computed **only from training (normal) data**:

threshold = 95th percentile of training reconstruction error

This prevents data leakage and ensures stable anomaly detection.

---

## 📊 Results

- **Accuracy:** ~78–80%
- **F1-score:** ~0.88

These results are consistent with standard unsupervised baselines used in the DCASE challenge.

---

## ▶️ How to Run

### 1️⃣ Clone the repository
```
git clone https://github.com/your-username/ai-ml-projects.git
cd ai-ml-projects/cnn-autoencoder-fan-anomaly-detection
```
### 2️⃣ Create data folder
```
  data/
   ├── dc2020t2l1-fan-train.npy
   ├── dc2020t2l1-fan-test.npy
   └── file_info.csv
```
### 3️⃣ Install dependencies
```
    pip install -r requirements.txt
```
### 4️⃣ Run training and evaluation
```
    python main.py
```
## 🔍 Key Highlights
Unsupervised learning (no anomaly labels during training)

CNN-based feature extraction

Autoencoder reconstruction modeling

Percentile-based thresholding

GPU-compatible PyTorch implementation

Suitable for predictive maintenance systems

## 🏭 Real-World Applications
Industrial machine monitoring

Predictive maintenance

Manufacturing quality control

Fault detection in rotating machinery

## 📘 Notes
This project focuses on modeling and anomaly detection logic, not deployment.

In production systems, inference is triggered automatically by sensor pipelines.

Manual user input is not required.

## 🚀 Future Improvements
Real-time inference pipeline

FastAPI deployment

Streaming audio processing

Transformer-based autoencoder comparison

Multimachine support

## 👤 Author
Santhosh
