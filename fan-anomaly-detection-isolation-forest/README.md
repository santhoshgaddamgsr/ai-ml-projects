# 🛠️ Fan Noise Anomaly Detection using Isolation Forest

This project implements an **acoustic-based anomaly detection system** designed for end-of-line quality inspection in manufacturing. By analyzing industrial fan audio features from the **DCASE 2020 Challenge**, the system automatically identifies defective units based on sound signatures.

---

## 📋 Project Overview

In smart manufacturing, defects often manifest as abnormal sound patterns. This system is designed to detect issues that are hard to see but easy to hear, such as:
* **Grinding or Rubbing** sounds in fan motors.
* **Clicking/Knocking** in mechanical gear units.
* **Irregular Airflow** or turbulence noise.
* **Structural Vibrations** caused by misaligned components.



---

## 📌 Problem Statement

The goal is to build a robust system that can:
**Automatically classify products as "Normal" or "Defective" using sound signals without manual human inspection.**

This project utilizes the **Isolation Forest** algorithm, which is highly effective at detecting anomalies by isolating outliers in high-dimensional feature data.

---

## 📂 Dataset & References

This project is based on the pre-processing work done in the DCASE 2020 Task 2 challenge.

* **Full Project Notebook:** [DCASE 2020 Task 2 - Preprocessed EDA](https://www.kaggle.com/code/muhammadmahtab/dcase-2020-task-2-preprocessed-eda/notebook)

### 1. Download Required Data
To run the code, download these specific files from the Kaggle project pages:

* **From the [Input Directory](https://www.kaggle.com/code/muhammadmahtab/dcase-2020-task-2-preprocessed-eda/input):**
    * `dc2020t2l1-fan-train.npy` (Training Features)
    * `dc2020t2l1-fan-test.npy` (Testing Features)
* **From the [Output Directory](https://www.kaggle.com/code/muhammadmahtab/dcase-2020-task-2-preprocessed-eda/output):**
    * `file_info.csv` (Labels & Metadata)

### 2. File Structure
Ensure your project folder is organized as follows:

```text
fan-anomaly-detection-isolation-forest/
 ├── main.py            # Main execution script
 ├── file_info.csv      # Ground-truth labels
 ├── dc2020t2l1-fan-train.npy
 ├── dc2020t2l1-fan-test.npy
 └── requirements.txt   # Dependencies
