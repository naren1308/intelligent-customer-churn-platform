import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Churn & Revenue Dashboard", layout="wide")

st.title("Intelligent Customer Churn & Revenue Platform")
st.markdown("### Powered by XGBoost & SHAP")

@st.cache_resource
def load_assets():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.abspath(os.path.join(current_dir, '../models/xgboost_model.pkl'))
    features_path = os.path.abspath(os.path.join(current_dir, '../models/xgboost_model_features.pkl'))
    
    if os.path.exists(model_path) and os.path.exists(features_path):
        model = joblib.load(model_path)
        feature_names = joblib.load(features_path)
        return model, feature_names
    return None, None

model, feature_names = load_assets()

if model is None:
    st.error("Model not found! Please run `python src/train.py` first.")
else:
    st.sidebar.header("Customer Input")
    
    input_data = {}
    st.sidebar.markdown("**Demographics**")
    input_data['city'] = st.sidebar.number_input("City ID", value=1)
    input_data['registered_via'] = st.sidebar.number_input("Registered Via", value=7)
    
    st.sidebar.markdown("**Financial / Payment**")
    input_data['CLV'] = st.sidebar.number_input("Customer Lifetime Value ($)", value=500.0)
    input_data['Total_Cancellations'] = st.sidebar.number_input("Total Cancellations", value=0)
    input_data['is_auto_renew'] = st.sidebar.selectbox("Auto Renew?", [1, 0])
    input_data['payment_plan_days'] = st.sidebar.number_input("Payment Plan Days", value=30)
    
    st.sidebar.markdown("**Engagement**")
    input_data['Avg_Daily_Listening_Secs'] = st.sidebar.number_input("Avg Daily Listening (Secs)", value=3600.0)
    input_data['Avg_Daily_Unique_Songs'] = st.sidebar.number_input("Avg Daily Unique Songs", value=20)
    
    st.sidebar.header("Business Logic")
    avg_customer_value = st.sidebar.number_input("Avg Customer Value ($)", value=141.69)
    
    if st.button("Analyze Risk"):
        # Fill missing features with 0 to match exactly what the model expects
        df_input = pd.DataFrame([input_data])
        for col in feature_names:
            if col not in df_input.columns:
                df_input[col] = 0.0
                
        # Reorder columns to match training
        df_input = df_input[feature_names]
        
        prob = model.predict_proba(df_input)[0][1]
        pred = model.predict(df_input)[0]
        
        # Determine risk color
        risk_color = "red" if prob > 0.5 else "green"
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"### Churn Probability: <span style='color:{risk_color}'>{prob:.1%}</span>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"### Prediction: {'Churn Risk' if pred == 1 else 'Safe'}")
        with col3:
            expected_saved = 1 * avg_customer_value if pred == 1 else 0
            st.markdown(f"### Revenue Saved: <span style='color:blue'>${expected_saved:.2f}</span>", unsafe_allow_html=True)
            
        st.markdown("---")
        st.markdown("### Why is this happening? (SHAP Explainability)")
        
        try:
            # Generate SHAP values
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(df_input)
            
            # Plot
            fig, ax = plt.subplots(figsize=(10, 4))
            shap.waterfall_plot(shap.Explanation(values=shap_values[0], 
                                                 base_values=explainer.expected_value, 
                                                 data=df_input.iloc[0], 
                                                 feature_names=feature_names), 
                                show=False)
            st.pyplot(fig)
        except Exception as e:
            st.warning("SHAP explanation failed. Please ensure 'shap' and 'matplotlib' are installed correctly.")
            st.write(str(e))
