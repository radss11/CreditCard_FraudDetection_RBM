# FraudLens AI — RBM-Based Credit Card Fraud Detection

An AI-powered anomaly detection system that identifies potentially fraudulent credit card transactions using a Restricted Boltzmann Machine (RBM).

FraudLens AI learns patterns from legitimate transactions and identifies unusual transactions through reconstruction error. It also provides an interactive Streamlit dashboard for transaction analysis and investigation.

## Project Overview

Credit card fraud detection is challenging because fraudulent transactions represent only a very small proportion of all transactions. Traditional classification approaches can struggle with this severe class imbalance.

FraudLens AI uses an **unsupervised anomaly detection approach**. The RBM is trained primarily on normal transactions and learns their underlying patterns. Transactions with unusually high reconstruction errors are flagged for further investigation.

## Key Features

- **RBM-based anomaly detection:** Learns patterns from legitimate transactions.
- **Unsupervised training:** Trains the model using normal transactions.
- **Reconstruction error scoring:** Assigns an anomaly score to each transaction.
- **Adjustable threshold:** Allows users to control anomaly sensitivity.
- **Interactive Streamlit dashboard:** Provides transaction analytics and visualizations.
- **CSV upload:** Supports scoring transaction datasets with the expected feature format.
- **Investigation queue:** Helps review potentially suspicious transactions.
- **CSV export:** Allows users to export scored and flagged transactions.

## Tech Stack

| Category | Technologies |
|---|---|
| Language | Python |
| Machine Learning | PyTorch |
| Data Processing | Pandas, NumPy |
| Preprocessing | Scikit-learn |
| Model | Restricted Boltzmann Machine (RBM) |
| Frontend / Dashboard | Streamlit |
| Visualization | Streamlit charts |
| Model Persistence | PyTorch, Joblib |

## Dataset

The project uses the Credit Card Fraud Detection dataset available on Kaggle.

- Total transactions: 284,807
- Legitimate transactions: 284,315
- Fraudulent transactions: 492
- Input features: 30
- Target column: `Class`

The dataset contains anonymized PCA-transformed features (`V1`–`V28`), along with `Time` and `Amount`.

**Note:** The dataset is not included in this repository. Download it from Kaggle and place `creditcard.csv` inside the `data/` directory.

## Model Architecture

The project implements a Restricted Boltzmann Machine using PyTorch.

The RBM consists of:

- Visible layer: 30 units
- Hidden layer: 64 units (extended experiment)
- Training algorithm: Contrastive Divergence (CD-1)
- Learning rate: 0.001
- Batch size: 256

The model learns the distribution of normal transactions and attempts to reconstruct its input.

### Anomaly Detection

The reconstruction error is calculated using Mean Squared Error (MSE):

\[
E(x)=\frac{1}{30}\sum_{i=1}^{30}(x_i-\hat{x}_i)^2
\]

Where:

- \(x_i\): Original input feature
- \(\hat{x}_i\): Reconstructed feature
- \(E(x)\): Reconstruction error

A transaction is flagged as potentially anomalous when:

\[
E(x) > \text{Threshold}
\]

A higher reconstruction error indicates that the transaction differs from the patterns learned by the model.

**Important:** An anomaly flag indicates a transaction requiring investigation; it does not confirm fraud.

## Project Workflow

1. Load the credit card transaction dataset.
2. Split the data into training, validation and testing sets.
3. Train the MinMaxScaler using normal training transactions.
4. Train the RBM using normal transactions.
5. Reconstruct transaction features and calculate reconstruction errors.
6. Select an anomaly threshold using validation data.
7. Score transactions using the trained model.
8. Display results through the Streamlit dashboard.

## Model Evaluation

The following results were obtained on the validation dataset during the extended training experiment.

| Metric | Validation Result |
|---|---:|
| ROC-AUC | 0.90795 |
| PR-AUC | 0.33897 |
| Precision | 46.67% |
| Recall | 47.30% |
| F1-Score | 0.4698 |

- Validation transactions: 42,721
- Fraudulent transactions: 74
- Selected threshold: 0.01937747

These are validation results, not held-out test results. The dataset is highly imbalanced, so precision, recall, F1-score and PR-AUC are important evaluation metrics alongside ROC-AUC.

## Project Structure

```text
CreditCard_FraudDetection_RBM/
│
├── app.py
├── prepare_data.py
├── train_rbm.py
├── evaluate_rbm.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── creditcard.csv
│   └── processed/
│
├── artifacts/
│   └── rbm_model.pt
│
└── outputs/
    └── training/
```

The `data/`, `artifacts/` and `outputs/` directories are generated locally and are not included in the GitHub repository.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/radss11/CreditCard_FraudDetection_RBM.git
cd CreditCard_FraudDetection_RBM
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Download the Credit Card Fraud Detection dataset from Kaggle and place `creditcard.csv` inside the `data/` folder.

### 5. Prepare the data

```bash
python prepare_data.py
```

### 6. Train the RBM

```bash
python train_rbm.py
```

### 7. Evaluate the model

```bash
python evaluate_rbm.py
```

### 8. Launch the dashboard

```bash
streamlit run app.py
```

**Note:** The model checkpoint and preprocessing artifacts must be generated before launching the dashboard.

## Limitations

- The model expects 30 features in the same PCA-transformed representation as the training dataset.
- Arbitrary raw banking transaction CSV files are not directly supported.
- Anomaly detection may produce false positives and false negatives.
- The current implementation is a prototype and is not intended for real-world financial deployment without further validation.

## Future Improvements

- Experiment with alternative anomaly detection models such as Autoencoders and Isolation Forest.
- Optimize anomaly thresholds using validation data.
- Improve fraud detection precision and recall.
- Evaluate performance on additional datasets.
- Explore real-time transaction monitoring.

## Author

**Radha Kamalwar**

B.Tech — Computer Science and Engineering (AI/ML)

GitHub: [@radss11](https://github.com/radss11)

---

*FraudLens AI — Detecting the Unusual. Enabling Smarter Investigation.*