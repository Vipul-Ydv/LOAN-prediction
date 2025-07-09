import joblib
import numpy as np

# Load models and encoders
def load_model(model_name='rf_model.joblib', encoder_name='label_encoder.joblib'):
    model = joblib.load(model_name)
    label_encoder = joblib.load(encoder_name)
    return model, label_encoder

# Get user input
def get_user_input():
    print('Enter applicant details:')
    age = int(input('Age (21-65): '))
    income = float(input('Annual Income (20000-150000): '))
    education = input('Education (High School/Bachelors/Masters/PhD): ')
    credit_score = int(input('Credit Score (300-850): '))
    loan_amount = float(input('Loan Amount (5000-50000): '))
    loan_term = int(input('Loan Term in months (12/24/36/48/60): '))
    employment_status = input('Employment Status (Employed/Self-Employed/Unemployed): ')
    marital_status = input('Marital Status (Single/Married/Divorced): ')
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

def main():
    model, label_encoder = load_model()
    user_data = get_user_input()
    # Convert to 2D array for model
    import pandas as pd
    X_input = pd.DataFrame([user_data])
    # Predict
    pred = model.predict(X_input)[0]
    proba = model.predict_proba(X_input)[0][1]  # Probability of approval
    result = label_encoder.inverse_transform([pred])[0]
    print(f'\nLoan Approval Prediction: {result}')
    print(f'Approval Confidence: {proba*100:.2f}%')

if __name__ == '__main__':
    main() 