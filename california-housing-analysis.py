import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# 1. Load and preprocess data
# Read the data
df = pd.read_csv('california_housing_test.csv')

# 2. Basic data exploration
print("Dataset Shape:", df.shape)
print("\nBasic Statistics:")
print(df.describe().round(2))

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 3. Data Preprocessing
# Create binary classification target based on median_house_value
median_price = df['median_house_value'].median()
df['price_category'] = (df['median_house_value'] > median_price).astype(int)

# Separate features and target
X = df.drop(['median_house_value', 'price_category'], axis=1)
y = df['price_category']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Feature Selection using ANOVA
# Perform ANOVA feature selection
f_scores, p_values = f_classif(X_train_scaled, y_train)
feature_scores = pd.DataFrame({
    'Feature': X.columns,
    'F_Score': f_scores,
    'P_value': p_values
})
feature_scores = feature_scores.sort_values('F_Score', ascending=False)
print("\nANOVA Feature Selection Results:")
print(feature_scores)

# 5. Visualization
plt.figure(figsize=(12, 6))
sns.barplot(x='Feature', y='F_Score', data=feature_scores)
plt.xticks(rotation=45)
plt.title('Feature Importance Scores (ANOVA)')
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

# Distribution of house values
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='median_house_value', bins=50)
plt.title('Distribution of House Values')
plt.show()

# 6. Classification using Random Forest
# Train Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_scaled, y_train)

# Make predictions
y_pred = rf_classifier.predict(X_test_scaled)

# Print classification report
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred))

# Feature importance from Random Forest
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_classifier.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nRandom Forest Feature Importance:")
print(feature_importance)

# Visualize Random Forest feature importance
plt.figure(figsize=(12, 6))
sns.barplot(x='Feature', y='Importance', data=feature_importance)
plt.xticks(rotation=45)
plt.title('Feature Importance (Random Forest)')
plt.tight_layout()
plt.show()

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
