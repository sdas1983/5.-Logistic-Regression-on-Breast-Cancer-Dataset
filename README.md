# 5. Logistic Regression on Breast Cancer Dataset

This project performs logistic regression on the Breast Cancer dataset to classify whether a tumor is malignant or benign based on various features. The dataset is sourced from [Deepak525's Breast Cancer Visualization and Classification GitHub repository](https://github.com/deepak525/Breast-Cancer-Visualization-and-Classification/blob/master/data.csv).

## Table of Contents

- [Data Overview](#data-overview)
- [Data Cleaning and Preparation](#data-cleaning-and-preparation)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Results and Visualization](#results-and-visualization)
- [License](#license)

## Data Overview

The dataset contains the following columns:
- `id`: Patient ID
- `diagnosis`: Diagnosis of breast cancer (M = malignant, B = benign)
- `radius_mean`: Mean radius
- `texture_mean`: Mean texture
- `perimeter_mean`: Mean perimeter
- `area_mean`: Mean area
- `smoothness_mean`: Mean smoothness
- `compactness_mean`: Mean compactness
- `concavity_mean`: Mean concavity
- `concave_points_mean`: Mean concave points
- `symmetry_mean`: Mean symmetry
- `fractal_dimension_mean`: Mean fractal dimension
- `radius_se`: Standard error of radius
- `texture_se`: Standard error of texture
- `perimeter_se`: Standard error of perimeter
- `area_se`: Standard error of area
- `smoothness_se`: Standard error of smoothness
- `compactness_se`: Standard error of compactness
- `concavity_se`: Standard error of concavity
- `concave_points_se`: Standard error of concave points
- `symmetry_se`: Standard error of symmetry
- `fractal_dimension_se`: Standard error of fractal dimension
- `radius_worst`: Worst radius
- `texture_worst`: Worst texture
- `perimeter_worst`: Worst perimeter
- `area_worst`: Worst area
- `smoothness_worst`: Worst smoothness
- `compactness_worst`: Worst compactness
- `concavity_worst`: Worst concavity
- `concave_points_worst`: Worst concave points
- `symmetry_worst`: Worst symmetry
- `fractal_dimension_worst`: Worst fractal dimension

## Data Cleaning and Preparation

1. **Data Loading**: Load the dataset from a CSV file.
2. **Data Inspection**: Check for missing values and basic statistics.
3. **Data Visualization**: 
   - Box plot of feature distributions.
4. **Feature and Target Preparation**:
   - Separate features (`X`) and target variable (`y`).
5. **Data Splitting**: Split the data into training and testing sets.
6. **Feature Scaling**: Standardize features using `StandardScaler`.

## Model Training and Evaluation

1. **Model Training**: Train a `LogisticRegression` model on the training data.
2. **Model Persistence**: Save the trained model using `pickle`.
3. **Prediction**: Predict using a sample input.
4. **Model Evaluation**:
   - Evaluate model accuracy on both training and testing sets.
   - Display a confusion matrix.
   - Generate a classification report.
   - Plot the ROC curve.

## Results and Visualization

- **Box Plot**: Visualizes the distribution of features.
- **Confusion Matrix**: Shows the performance of the model in classifying the test data.
- **Classification Report**: Provides precision, recall, and F1-score for each class.
- **ROC Curve**: Visualizes the ROC curve for evaluating the model's performance.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
