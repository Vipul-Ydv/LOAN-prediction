import streamlit as st
import pandas as pd
import joblib

# Load models and encoders
def load_model(model_name='rf_model.joblib', encoder_name='label_encoder.joblib'):
    model = joblib.load(model_name)
    label_encoder = joblib.load(encoder_name)
    return model, label_encoder

st.title('Loan Eligibility Predictor')
st.write('Enter applicant details to predict loan approval:')

# Input form
def user_input_form():
    age = st.number_input('Age', min_value=21, max_value=65, value=30)
    income = st.number_input('Annual Income', min_value=20000, max_value=150000, value=50000)
    education = st.selectbox('Education', ['High School', 'Bachelors', 'Masters', 'PhD'])
    credit_score = st.number_input('Credit Score', min_value=300, max_value=850, value=650)
    loan_amount = st.number_input('Loan Amount', min_value=5000, max_value=50000, value=20000)
    loan_term = st.selectbox('Loan Term (months)', [12, 24, 36, 48, 60])
    employment_status = st.selectbox('Employment Status', ['Employed', 'Self-Employed', 'Unemployed'])
    marital_status = st.selectbox('Marital Status', ['Single', 'Married', 'Divorced'])
    return {
        'age': age,
        'income': income,
        'education': education,
        'credit_score': credit_score,
        'loan_amount': loan_amount,
        'loan_term': loan_term,
        'employment_status': employment_status,
        'marital_status': marital_status
    }

model, label_encoder = load_model()

with st.form('loan_form'):
    user_data = user_input_form()
    submitted = st.form_submit_button('Predict Loan Approval')

if submitted:
    X_input = pd.DataFrame([user_data])
    pred = model.predict(X_input)[0]
    proba = model.predict_proba(X_input)[0][1]
    result = label_encoder.inverse_transform([pred])[0]
    st.subheader('Prediction Result')
    st.write(f'**Loan Approval Prediction:** {result}')
    st.write(f'**Approval Confidence:** {proba*100:.2f}%') 