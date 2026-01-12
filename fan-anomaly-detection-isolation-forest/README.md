# 🛠️ Fan Noise Anomaly Detection using Isolation Forest

This project implements an **acoustic-based anomaly detection system** designed for end-of-line quality inspection in manufacturing. By analyzing industrial fan audio features from the **DCASE 2020 Challenge**, the system automatically identifies defective units based on sound signatures.

---

## 📋 Project Overview

In smart manufacturing, defects often manifest as abnormal sound patterns. This system identifies mechanical issues that are difficult to see but easy to detect through sound:

* **Grinding or Rubbing** sounds in fan motors.
* **Clicking/Knocking** in mechanical gear units.
* **Irregular Airflow** or turbulence noise.
* **Structural Vibrations** caused by misaligned components.



---

## 📌 Problem Statement

At the end of a production line, products must be classified as **Normal** or **Defective** before shipping. This project solves this by:
1. Using fan noise recordings from the **DCASE industrial dataset**.
2. Applying an **Isolation Forest** model to isolate abnormal acoustic behavior without requiring manual inspection.

---

## 📂 Dataset & References

This project is based on the pre-processing work from the DCASE 2020 Task 2 challenge.

* **Full Project Notebook:** [DCASE 2020 Task 2 - Preprocessed EDA](https://www.kaggle.com/code/muhammadmahtab/dcase-2020-task-2-preprocessed-eda/notebook)

### 1. Download Required Data
Download these specific files from the Kaggle project pages and place them in your project folder:

* **From the [Input Directory](https://www.kaggle.com/code/muhammadmahtab/dcase-2020-task-2-preprocessed-eda/input):**
    * `dc2020t2l1-fan-train.npy`
    * `dc2020t2l1-fan-test.npy`
* **From the [Output Directory](https://www.kaggle.com/code/muhammadmahtab/dcase-2020-task-2-preprocessed-eda/output):**
    * `file_info.csv`

### 2. Project Structure
```text
fan-anomaly-detection-isolation-forest/
 ├── main.py                # Training, evaluation & prediction
 ├── file_info.csv          # Ground-truth labels & metadata
 ├── dc2020t2l1-fan-train.npy
 ├── dc2020t2l1-fan-test.npy
 ├── requirements.txt       # Dependencies
 └── README.md 
```

## ▶️ How to Run

1. **Install Dependencies**  
Ensure you have Python installed, then run:  
pip install -r requirements.txt  

2. **Run the Model**  
Execute the main script to train the model and view results:  
python main.py  

The script will perform the following:  
- Load and flatten the train and test audio features from the .npy files  
- Perform a hyperparameter grid search for contamination and n_estimators  
- Print the best F1 score achieved  
- Display the first 20 anomaly predictions  

---

## 🎯 What this demonstrates

- **Industrial Audio Detection** – Real-world application of machine learning in manufacturing quality control  
- **Feature-based Pipeline** – Efficient handling and flattening of high-dimensional .npy audio data  
- **Model Evaluation** – Use of the **F1 score** to measure performance on imbalanced anomaly datasets  
- **Baseline Foundation** – A strong baseline for advancing to deep learning models such as CNN Autoencoders  
