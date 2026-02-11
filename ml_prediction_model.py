# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

# ----------------------------------------------
# 1. Load Dataset
# ----------------------------------------------
print("Loading Dataset...")

df = pd.read_csv("churn_data.csv")  # Place dataset in same folder

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

# ----------------------------------------------
# 2. Data Preprocessing
# ----------------------------------------------
print("\nPreprocessing Data...")

# Drop CustomerID if exists
if "CustomerID" in df.columns:
    df.drop("CustomerID", axis=1, inplace=True)

# Convert TotalCharges to numeric if exists
if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Fill missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

# Encode categorical variables
label_encoder = LabelEncoder()

for column in df.select_dtypes(include=['object']).columns:
    df[column] = label_encoder.fit_transform(df[column])

print("\nPreprocessing Completed!")

# ----------------------------------------------
# 3. Exploratory Data Analysis
# ----------------------------------------------
print("\nGenerating Visualizations...")

plt.figure(figsize=(6,4))
sns.countplot(x="Churn", data=df)
plt.title("Churn Distribution")
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# ----------------------------------------------
# 4. Feature Selection
# ----------------------------------------------
X = df.drop("Churn", axis=1)
y = df["Churn"]

# ----------------------------------------------
# 5. Train Test Split
# ----------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# ----------------------------------------------
# 6. Model Training
# ----------------------------------------------
print("\nTraining Random Forest Model...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("Model Training Completed!")

# ----------------------------------------------
# 7. Model Evaluation
# ----------------------------------------------
print("\nEvaluating Model...")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")
print("ROC-AUC Score:", round(roc_auc, 3))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ----------------------------------------------
# 8. Confusion Matrix
# ----------------------------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ----------------------------------------------
# 9. ROC Curve
# ----------------------------------------------
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label="ROC Curve")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# ----------------------------------------------
# 10. Feature Importance
# ----------------------------------------------
feature_importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

plt.figure(figsize=(10,6))
feature_importance.plot(kind="bar")
plt.title("Feature Importance")
plt.show()

# ----------------------------------------------
# 11. Save Model
# ----------------------------------------------
joblib.dump(model, "churn_model.pkl")

print("\nModel Saved Successfully as 'churn_model.pkl'")

print("\nProject Execution Completed Successfully!")