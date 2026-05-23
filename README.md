# Intelligent Customer Churn + Revenue Retention Platform 📊

An end-to-end, industry-aligned machine learning pipeline that goes beyond predicting customer churn. This platform processes raw, relational streaming data to actively flag at-risk customers, explain *why* they are leaving using **SHAP**, and calculate the direct **Financial Revenue Impact** of retaining them.

## 🚀 The Business Problem
Most tutorial churn models stop at "92% Accuracy" using synthetic or clean datasets. Industry teams ask: *"How much money does this save?"*

This platform answers that question by:
1. Identifying high-risk customers before they cancel.
2. Explaining the hidden reasons behind their behavior.
3. Translating those probabilities into a concrete "Expected Revenue Saved" metric.

## 💾 Dataset: KKBox Music Streaming
We avoid synthetic "toy" datasets. This project is built on the massive [WSDM KKBox Churn Prediction Dataset](https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge). It features heavily relational data requiring complex pandas joins across:
*   **User Demographics** (`members.csv`)
*   **Historical Transactions** (`transactions.csv`)
*   **Daily Listening Logs** (`user_logs.csv` - originally ~30GB uncompressed)

## 🏗️ Architecture & Pipeline

### 1. Data Engineering (`src/preprocess.py`)
Extracts signal from noise by aggregating gigabytes of relational tables to engineer complex features:
*   **Customer Lifetime Value (CLV)**
*   **Historical Cancellations**
*   **Average Daily Listening Seconds**

### 2. Advanced Modeling (`src/train.py`)
*   **Algorithm**: `XGBoost` Classifier.
*   **Imbalance Handling**: Dynamic `scale_pos_weight` calculation to ensure the model catches the minority "Churn" class.
*   **Evaluation**: ROC-AUC scoring, optimized for real-world predictive power.

### 3. Business & Explainability Dashboard (`dashboard/app.py`)
*   **Streamlit UI**: An interactive frontend to test individual customer risk.
*   **SHAP Integration**: Visualizes exactly which features (e.g., dropping usage or turning off auto-renew) are driving the churn prediction via Waterfall plots.
*   **Financial Impact**: Embeds the global Average Customer Value (`$141.69`) to output real dollar amounts.

---

## 🛠️ Tech Stack
*   **Data Processing**: `pandas`, `numpy`, `py7zr`
*   **Machine Learning**: `scikit-learn`, `xgboost`
*   **Explainable AI**: `shap`, `matplotlib`
*   **Deployment**: `streamlit`

---

## 💻 How to Run Locally

### 1. Setup Environment
```bash
# Clone the repository
git clone https://github.com/naren1308/intelligent-customer-churn-platform.git
cd intelligent-customer-churn-platform

# Install dependencies
pip install -r requirements.txt
```

### 2. Procure the Data
Download the KKBox dataset from Kaggle and place the `.7z` archives in the `data/` folder.
```bash
# Extract the compressed data
python src/extract.py
```

### 3. Execute the Pipeline
Run the data engineering and training scripts in sequence:
```bash
# Aggregate relational tables and generate features
python src/preprocess.py

# Train the XGBoost model and save artifacts
python src/train.py
```

### 4. Launch the Dashboard
```bash
streamlit run dashboard/app.py
```
