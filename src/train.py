import pandas as pd
import os
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

def train_model(data_path, model_path):
    print(f"Loading processed data from {data_path}...")
    df = pd.read_csv(data_path)
    
    y = df['is_churn']
    X = df.drop(columns=['is_churn'])
    
    # Train test split to evaluate properly
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Handle Class Imbalance with scale_pos_weight
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train) if sum(y_train) > 0 else 1
    
    print("Training XGBoost Model...")
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    print("Evaluating Model...")
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    print("ROC-AUC Score:", roc_auc_score(y_test, probs))
    print(classification_report(y_test, preds))
    
    print(f"Saving model to {model_path}...")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    
    # Save the feature names for SHAP and Dashboard
    joblib.dump(list(X.columns), model_path.replace('.pkl', '_features.pkl'))
    print("Model saved successfully!")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.abspath(os.path.join(current_dir, '../data/processed_train.csv'))
    MODEL_PATH = os.path.abspath(os.path.join(current_dir, '../models/xgboost_model.pkl'))
    
    train_model(DATA_PATH, MODEL_PATH)
