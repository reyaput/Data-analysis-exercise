from matplotlib import cm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score)

id_url = '196eYlk3aAtUCGhzhc23akN5B4cznRHik'
url_csv = f'https://drive.google.com/uc?export=download&id={id_url}'
df = pd.read_csv(url_csv)

original_features = [
    'CreditScore', 'Age', 'Tenure', 'Balance', 
    'NumOfProducts', 'HasCrCard', 'IsActiveMember', 
    'EstimatedSalary', 'Complain', 'Satisfaction Score', 
    'Point Earned'
]

categorical_features = ['Geography', 'Gender', 'Card Type']

engineered_features = [
    'AgeGroup', 'Tenure_Category', 
    'Balance_Category', 'CreditScore_Category'
]

df_model = df.copy()

label_encoders = {}

categorical_cols = ['Geography', 'Gender', 'Card Type', 
                   'AgeGroup', 'Tenure_Category', 
                   'Balance_Category', 'CreditScore_Category']

for col in categorical_cols:
    le = LabelEncoder()
    df_model[col + '_Encoded'] = le.fit_transform(df_model[col])
    label_encoders[col] = le


feature_list = [
    'CreditScore', 'Age', 'Tenure', 'Balance', 
    'NumOfProducts', 'HasCrCard', 'IsActiveMember', 
    'EstimatedSalary', 'Complain', 'Satisfaction Score', 
    'Point Earned',
    'Geography_Encoded', 'Gender_Encoded', 'Card Type_Encoded',
    'AgeGroup_Encoded', 'Tenure_Category_Encoded', 
    'Balance_Category_Encoded', 'CreditScore_Category_Encoded'
]

X = df_model[feature_list]
y = df_model['Exited']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_list)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_list)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

rf_model.fit(X_train, y_train)

y_pred_train = rf_model.predict(X_train)
y_pred_test = rf_model.predict(X_test)

y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)
precision = precision_score(y_test, y_pred_test)
recall = recall_score(y_test, y_pred_test)
f1 = f1_score(y_test, y_pred_test)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("="*60)
print("MODEL PERFORMANCE METRICS")
print("="*60)
print(f"Training Accuracy:   {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"Testing Accuracy:    {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"Precision:           {precision:.4f}")
print(f"Recall:              {recall:.4f}")
print(f"F1-Score:            {f1:.4f}")
print(f"ROC-AUC Score:       {roc_auc:.4f}")
print("="*60)

feature_importance = pd.DataFrame({
    'Feature': feature_list,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

# Visualize top 15 features
plt.figure(figsize=(12, 8))
top_features = feature_importance.head(15)
sns.barplot(data=top_features, y='Feature', x='Importance', palette='viridis')
plt.title('Top 15 Most Important Features', fontsize=14, fontweight='bold')
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.tight_layout()
plt.show()


joblib.dump(rf_model, 'random_forest_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')


