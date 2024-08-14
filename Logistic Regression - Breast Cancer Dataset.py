# Logistic Regression on Breast Cancer Dataset

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, classification_report, ConfusionMatrixDisplay, RocCurveDisplay
import pickle

# Set pandas display options
pd.set_option('display.max_columns', None)

# Load the dataset
# Data source: https://raw.githubusercontent.com/deepak525/Breast-Cancer-Visualization-and-Classification/master/data.csv
df = pd.read_csv(r"C:\Users\das.su\OneDrive - GEA\Documents\PDF\Machine Learning\BIT ML, AI and GenAI Course\Breast_Cancer_Data.csv")

# Display basic information about the dataset
print("Dataset Columns:")
print(df.columns)

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Description:")
print(df.describe())

print("\nMissing Values in Each Column:")
print(df.isnull().sum())

print("\nDataset Info:")
print(df.info())

# Visualization
# Uncomment the following lines to generate plots

# Pair Plot of Variables
# sns.pairplot(df, hue='diagnosis', kind='scatter')
# plt.show()

# Box Plot of Variables
plt.figure(figsize=(15, 7))
sns.boxplot(data=df.iloc[:, 2:12])
plt.title("Box Plot of Features")
plt.show()

# Preparing for Model Building
X = df.drop(['diagnosis'], axis=1)
y = df['diagnosis']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)  # Apply the same transformation to test data

# Train Logistic Regression Model
regression = LogisticRegression()
regression.fit(X_train, y_train)

# Save the model using pickle
pickle.dump(regression, open('Breast_Cancer_Logistic_model.pickle', 'wb'))

# Predict using a sample input
test_sample = scaler.transform([[0, 17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]])
prediction = regression.predict(test_sample)
print("\nSample Prediction:", prediction)

# Model Evaluation
print("\nModel Accuracy on Test Data:")
print(regression.score(X_test, y_test))

print("\nModel Accuracy on Training Data:")
print(regression.score(X_train, y_train))

# Confusion Matrix
print("\nConfusion Matrix:")
ConfusionMatrixDisplay.from_estimator(regression, X_test, y_test)
plt.title("Confusion Matrix")
plt.show()

# Classification Report
print("\nClassification Report:")
y_pred = regression.predict(X_test)
print(classification_report(y_test, y_pred))

# ROC Curve
print("\nROC Curve:")
RocCurveDisplay.from_estimator(regression, X_test, y_test)
plt.title("ROC Curve")
plt.show()
