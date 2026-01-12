Fan Noise Anomaly Detection using Isolation Forest

This project implements a noise-based anomaly detection system for end-of-line manufacturing quality inspection using industrial fan audio features from the DCASE 2020 Challenge.

In manufacturing environments, products such as cooling fans, electric motors, pumps, and small mechanical assemblies (e.g., toy cars, gear units) are tested at the end of the production line.

Defects in these products often appear as abnormal acoustic patterns, including:

	Grinding or rubbing sounds in motors and fans
	Clicking or knocking in gear mechanisms
	Irregular airflow or turbulence noise
	Vibrations caused by misaligned or worn components

These faults are difficult to detect visually but can be clearly identified through sound.

📌 Problem Statement

At the end of a manufacturing line, each product (fan, motor, pump, or mechanical unit) must be classified as normal or defective before it is shipped.

The challenge is to build a system that can:

	Automatically detect faulty products using only their sound signals, without requiring manual inspection.

This project solves this problem by using fan noise recordings from the DCASE industrial dataset and applying an Isolation Forest anomaly detection model to identify abnormal acoustic behavior.

The goal is to:
Automatically identify defective products using only their sound signatures.

📂 Dataset
This project is based on the DCASE 2020 Task 2 – Industrial Sound Dataset (Fan category).

DCASE Challenge & Preprocessing
https://www.kaggle.com/code/muhammadmahtab/dcase-2020-task-2-preprocessed-eda

Pre-extracted feature files
These files contain audio features extracted from fan sound recordings.

Train data
dc2020t2l1-fan-train.npy

Download:
https://storage.googleapis.com/kaggle-data-sets/562906/1025648/compressed/dc2020t2l1-fan-train.npy.zip

Test data
dc2020t2l1-fan-test.npy

Download:
https://storage.googleapis.com/kaggle-data-sets/562906/1025648/compressed/dc2020t2l1-fan-test.npy.zip

📑 Labels and Metadata
The file file_info.csv contains:
File names
Device type (fan, pump, etc.)
Train/Test split
Ground truth (normal / anomaly)

It is used to:
Extract true labels for fan test samples
Compute F1 score for model evaluation

CSV source:
https://storage.googleapis.com/kaggle-script-versions/216835682/output/file_info.csv

🧠 Model & Approach
This project uses:
Isolation Forest (unsupervised anomaly detection)
Feature vectors extracted from fan audio signals
Grid search on:
	contamination
	number of trees (n_estimators)

The model:
Trains only on normal fan sounds
Learns the pattern of healthy machines
Flags deviations as anomalies

📊 Evaluation
The model is evaluated using:
F1 Score
Based on ground-truth labels from file_info.csv

The script searches for the best hyperparameters and prints:
F1 score for each configuration
Best parameters
Final predictions

▶ How to Run
1. Download the data
Download and unzip:
	dc2020t2l1-fan-train.npy
	dc2020t2l1-fan-test.npy

Place them inside:
fan-anomaly-detection-isolation-forest/

2. Install dependencies
pip install -r requirements.txt

3. Run the model
python main.py


The script will:

Load train & test features

Train Isolation Forest

Perform hyperparameter search

Print the best F1 score

Display anomaly predictions

📁 Project Structure
fan-anomaly-detection-isolation-forest/
 ├── main.py          # Training, evaluation & prediction
 ├── file_info.csv    # Ground-truth labels & metadata
 ├── README.md
 └── requirements.txt

🎯 What this demonstrates

This project shows:
Industrial audio anomaly detection
Use of real manufacturing datasets (DCASE)
Feature-based ML pipeline
Model evaluation using F1 score
End-to-end reproducible experiment

This is a baseline industrial anomaly detection system and serves as a foundation for more advanced models such as CNN Autoencoders and deep learning-based acoustic models.