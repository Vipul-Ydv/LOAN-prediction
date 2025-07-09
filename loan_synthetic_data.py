import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

N = 1000  # Number of samples

def random_education(size):
    return np.random.choice(['High School', 'Bachelors', 'Masters', 'PhD'], size=size)

def random_employment(size):
    return np.random.choice(['Employed', 'Self-Employed', 'Unemployed'], size=size)

def random_marital(size):
    return np.random.choice(['Single', 'Married', 'Divorced'], size=size)

# Generate features
data = {
    'age': np.random.randint(21, 65, size=N),
    'income': np.random.randint(20000, 150000, size=N),
    'education': random_education(N),
    'credit_score': np.random.randint(300, 850, size=N),
    'loan_amount': np.random.randint(5000, 50000, size=N),
    'loan_term': np.random.choice([12, 24, 36, 48, 60], size=N),
    'employment_status': random_employment(N),
    'marital_status': random_marital(N),
}

df = pd.DataFrame(data)

# Simple rule-based target for demonstration
# Approve if income > 40k, credit_score > 600, and not unemployed
conditions = (
    (df['income'] > 40000) &
    (df['credit_score'] > 600) &
    (df['employment_status'] != 'Unemployed')
)
df['loan_approved'] = np.where(conditions, 'Yes', 'No')

# Introduce some missing values randomly
for col in ['education', 'credit_score', 'marital_status']:
    df.loc[df.sample(frac=0.05).index, col] = np.nan

# Save to CSV
df.to_csv('loan_synthetic_data.csv', index=False)

print('Synthetic dataset generated and saved as loan_synthetic_data.csv') 