import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin

# Настройка страницы
st.set_page_config(page_title="Fraud Detection App", page_icon="💳", layout="wide")

# --- 1. ВСТАВЛЯЕМ КАСТОМНЫЕ КЛАССЫ ИЗ ВАШЕГО НОУТБУКА ---
class CyclicalTimeTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = np.asarray(X).ravel()
        hour = (X % 86400) / 3600
        sin_h = np.sin(2 * np.pi * hour / 24).reshape(-1, 1)
        cos_h = np.cos(2 * np.pi * hour / 24).reshape(-1, 1)
        return np.hstack([sin_h, cos_h])
    def get_feature_names_out(self, input_features=None):
        return np.array(['hour_sin', 'hour_cos'])

class LogTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return np.log1p(np.asarray(X))
    def get_feature_names_out(self, input_features=None):
        return np.array(['log_Amount'])

class InteractionAdder(BaseEstimator, TransformerMixin):
    def __init__(self, v17_idx=16, log_amt_idx=-1):
        self.v17_idx = v17_idx
        self.log_amt_idx = log_amt_idx
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = np.asarray(X)
        interaction = (X[:, self.v17_idx] * X[:, self.log_amt_idx]).reshape(-1, 1)
        return np.hstack([X, interaction])
# --------------------------------------------------------

st.title("💳 Credit Card Fraud Detection")
st.write("""
This application predicts the probability of a transaction being fraudulent. 
Adjust the most critical PCA features and the transaction amount in the sidebar to see how the model's decision changes in real-time.
""")

# 2. Загрузка модели
@st.cache_resource
def load_model():
    return joblib.load('fraud_model.pkl')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}. Please ensure 'fraud_model.pkl' is in the directory.")
    st.stop()

# 3. Боковая панель ввода
st.sidebar.header("⚙️ Transaction Features")
st.sidebar.write("Adjust the most impactful features (based on SHAP analysis):")

def user_input_features():
    v14 = st.sidebar.slider('V14 (Strongest Fraud Signal)', -20.0, 10.0, -5.0)
    v12 = st.sidebar.slider('V12', -15.0, 10.0, -2.0)
    v17 = st.sidebar.slider('V17', -15.0, 10.0, -3.0)
    v10 = st.sidebar.slider('V10', -15.0, 10.0, -1.0)
    v4  = st.sidebar.slider('V4', -5.0, 15.0, 2.0)
    amount = st.sidebar.number_input('Transaction Amount ($)', min_value=0.0, max_value=25000.0, value=150.0)
    time = st.sidebar.number_input('Time (Seconds since start)', min_value=0, value=3600)
    
    data = {'Time': time}
    for i in range(1, 29):
        data[f'V{i}'] = 0.0
    data['Amount'] = amount
    
    data['V14'] = v14
    data['V12'] = v12
    data['V17'] = v17
    data['V10'] = v10
    data['V4'] = v4
    
    columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    df = pd.DataFrame(data, index=[0])[columns]
    return df

input_df = user_input_features()

# 4. Кнопка предсказания
if st.button("🔍 Predict Fraud Risk", type="primary"):
    with st.spinner('Analyzing transaction...'):
        proba = model.predict_proba(input_df)[0][1]
        
        st.markdown("---")
        st.subheader("Prediction Result")
        
        col1, col2 = st.columns([1, 2])
        with col1:
            if proba > 0.5:
                st.error(f"⚠️ HIGH RISK")
                st.metric(label="Fraud Probability", value=f"{proba*100:.1f}%")
            else:
                st.success(f"✅ LOW RISK")
                st.metric(label="Fraud Probability", value=f"{proba*100:.1f}%")

        # График SHAP
        with col2:
            st.subheader("Decision Explanation (SHAP)")
            st.write("This waterfall plot shows exactly which features pushed the risk score up (red) or down (blue).")
            
            try:
                clf = model.named_steps['clf'] 
                preprocessed_input = model[:-1].transform(input_df)
                
                explainer = shap.TreeExplainer(clf)
                shap_values = explainer(preprocessed_input)
                
                fig = plt.figure(figsize=(8, 4))
                if len(shap_values.shape) == 3:
                    shap.plots.waterfall(shap_values[0, :, 1], show=False, max_display=10)
                else:
                    shap.plots.waterfall(shap_values[0], show=False, max_display=10)
                
                st.pyplot(fig)
                plt.clf()
            except Exception as e:
                st.warning(f"Could not generate SHAP explanation. Error: {e}")