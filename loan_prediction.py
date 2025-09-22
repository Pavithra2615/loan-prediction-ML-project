# -------------------------------
# Loan Prediction ML Project
# -------------------------------

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Step 2: Load Dataset
data = pd.read_csv("loan.csv")   # <-- Make sure loan.csv is in same folder
print("Dataset Shape:", data.shape)
print(data.head())

# Step 3: Handle Missing Values
data['LoanAmount'] = data['LoanAmount'].fillna(data['LoanAmount'].mean())
data['Loan_Amount_Term'] = data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mode()[0])
data['Credit_History'] = data['Credit_History'].fillna(data['Credit_History'].mode()[0])

# Step 4: Encode Categorical Variables
encoder = LabelEncoder()
for col in ['Gender','Married','Education','Self_Employed','Property_Area','Loan_Status']:
    data[col] = encoder.fit_transform(data[col])

print("\nAfter Encoding:\n", data.head())

# Step 5: Split Features & Target
X = data.drop(columns=['Loan_Status','Loan_ID'])
y = data['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

print("\n🔹 Logistic Regression Results")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))

# Step 7: Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("\n🔹 Random Forest Results")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# Step 8: Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Step 9: Predict for New Applicant
# Format: ['Gender','Married','Dependents','Education','Self_Employed',
#          'ApplicantIncome','CoapplicantIncome','LoanAmount',
#          'Loan_Amount_Term','Credit_History','Property_Area']

new_applicant = np.array([[1,1,0,1,0,5000,2000,150,360,1,2]])
prediction = rf_model.predict(new_applicant)

print("\nNew Applicant Prediction:")
if prediction[0] == 1:
    print("Loan Approved")
else:
    print("Loan Not Approved")
