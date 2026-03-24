import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

#Load loan dataset from kaggle
print("Loading loan data from Kaggle...")
df = pd.read_csv('cs-training.csv')

print(f"✅ Loaded {len(df)} borrower records")
print(f"📋 Columns: {df.columns.tolist()}\n")

#Data cleaning:
# Drop the first column (Unnamed: 0) - it's just an index
df = df.drop(columns=['Unnamed: 0'], errors='ignore')

#Rename target column to something simpler
df = df.rename(columns={'SeriousDlqin2yrs': 'default'})

#Check for missing values
print(f"Missing values before cleaning:\n{df.isnull().sum()}\n")

# Fill missing MonthlyIncome with median
df['MonthlyIncome'].fillna(df['MonthlyIncome'].median(), inplace=True)

# Fill missing NumberOfDependents with 0
df['NumberOfDependents'].fillna(0, inplace=True)

print("Missing values handled\n")

#Features 
feature_cols = [col for col in df.columns if col != 'default']
X = df[feature_cols]
y = df['default']

print(f"Features: {feature_cols}")
print(f"Default rate: {y.mean():.2%}\n")

#Train model

print("Training Random Forest Classifier")

#Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2%}\n")

# Feature importance to show which factors drive the risk of divorce

print("Feature Importance (what drives default risk):")
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for i, row in importance.iterrows():
    print(f"   {row['feature']}: {row['importance']:.1%}")

#Visualisation

#1. Feature Importance Chart
plt.figure(figsize=(10, 6))
plt.barh(importance['feature'], importance['importance'])
plt.xlabel('Importance')
plt.title('Loan Default Model: Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()

#2. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()

#3. Default Rate by Age
plt.figure(figsize=(10, 6))
age_bins = pd.cut(df['age'], bins=[0, 30, 40, 50, 60, 100])
default_by_age = df.groupby(age_bins)['default'].mean()
default_by_age.plot(kind='bar', color='coral')
plt.title('Default Rate by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Default Rate')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('default_by_age.png')
plt.show()

print("\nLoan Default Model Complete!")

# Interactive prediction

print("\n" + "="*50)
print("TRY IT YOURSELF: Will you default?")
print("="*50)

# Get user input
print("\nEnter your financial info:")
credit_util = float(input("Credit utilization (0.0 to 1.0): "))
debt_ratio = float(input("Debt to income ratio (0.0 to 2.0): "))
monthly_income = float(input("Monthly income ($): "))
age = int(input("Age: "))
late_90_days = int(input("Times 90+ days late in past: "))
late_30_59 = int(input("Times 30-59 days late: "))
late_60_89 = int(input("Times 60-89 days late: "))
open_credit = int(input("Number of open credit lines: "))
real_estate = int(input("Number of real estate loans: "))
dependents = int(input("Number of dependents: "))

# Create input array
user_data = pd.DataFrame([[
    credit_util, age, late_30_59, debt_ratio, monthly_income,
    open_credit, late_90_days, real_estate, late_60_89, dependents
]], columns=feature_cols)

# Predict
prediction = model.predict(user_data)[0]
probability = model.predict_proba(user_data)[0][1]

# Show result
print("\n" + "="*50)
print("PREDICTION RESULT")
print("="*50)

if prediction == 1:
    print(f"⚠️ HIGH RISK: {probability:.1%} chance of default")
    print("   Recommendation: Review credit usage and debt ratio")
else:
    print(f"🟢 LOW RISK: {(1-probability):.1%} chance of paying back")
    print("   Good financial profile")

print("="*50)