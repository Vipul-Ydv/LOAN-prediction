import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, RocCurveDisplay
import matplotlib.pyplot as plt
import joblib

# Load dataset
df = pd.read_csv('loan_synthetic_data.csv')

# Separate features and target
y = df['loan_approved']
X = df.drop('loan_approved', axis=1)

# Identify categorical and numerical columns
categorical_cols = ['education', 'employment_status', 'marital_status']
numerical_cols = ['age', 'income', 'credit_score', 'loan_amount', 'loan_term']

# Preprocessing for numerical data: impute missing values, scale
numerical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data: impute missing, one-hot encode
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessors
preprocessor = ColumnTransformer([
    ('num', numerical_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# Encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # Yes=1, No=0

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Pipelines for models
logreg_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train models
logreg_pipeline.fit(X_train, y_train)
rf_pipeline.fit(X_train, y_train)

# Predict
y_pred_logreg = logreg_pipeline.predict(X_test)
y_pred_rf = rf_pipeline.predict(X_test)

# Probabilities for ROC
y_proba_logreg = logreg_pipeline.predict_proba(X_test)[:, 1]
y_proba_rf = rf_pipeline.predict_proba(X_test)[:, 1]

# Evaluation
print('Logistic Regression Classification Report:')
print(classification_report(y_test, y_pred_logreg))
print('Random Forest Classification Report:')
print(classification_report(y_test, y_pred_rf))

print('Logistic Regression Confusion Matrix:')
print(confusion_matrix(y_test, y_pred_logreg))
print('Random Forest Confusion Matrix:')
print(confusion_matrix(y_test, y_pred_rf))

# ROC Curve
fpr_logreg, tpr_logreg, _ = roc_curve(y_test, y_proba_logreg)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
roc_auc_logreg = auc(fpr_logreg, tpr_logreg)
roc_auc_rf = auc(fpr_rf, tpr_rf)

plt.figure(figsize=(8, 6))
plt.plot(fpr_logreg, tpr_logreg, label=f'Logistic Regression (AUC = {roc_auc_logreg:.2f})')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_rf:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.tight_layout()
plt.savefig('roc_curve.png')
plt.show()

# Save models and encoders
joblib.dump(logreg_pipeline, 'logreg_model.joblib')
joblib.dump(rf_pipeline, 'rf_model.joblib')
joblib.dump(label_encoder, 'label_encoder.joblib')

print('Models and encoders saved. ROC curve plotted as roc_curve.png.') 