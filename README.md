# Cognizant Artificial Intelligence Job Simulation on Forage - May 2024

![Cognizant](https://github.com/RSN601KRI/cognizant_ai_virtual_internship/assets/106860359/4960c997-2453-43b1-b5a1-e59f3871cbe3)

## Overview

This README file provides an overview of the Cognizant Artificial Intelligence Job Simulation completed on Forage in May 2024. The simulation was designed to provide hands-on experience with AI-related tasks relevant to Cognizant's Data Science team. The main activities included exploratory data analysis, model training, and presenting findings to the business.

## Objectives

- Conduct exploratory data analysis (EDA) for a technology-led client, Gala Groceries.
- Prepare a Python module for model training and performance evaluation.
- Communicate findings and analysis through a PowerPoint presentation.

## Tasks Completed

### 1. Exploratory Data Analysis (EDA)

**Tools Used:**
- Python
- Google Colab

**Description:**
Conducted EDA on a dataset provided by Gala Groceries to uncover patterns, insights, and relationships within the data. This involved:
- Cleaning and preprocessing the data.
- Visualizing data distributions and relationships.
- Identifying key features and potential areas for further analysis.

**Key Steps:**
- Imported necessary libraries (pandas, numpy, matplotlib, seaborn).
- Loaded and explored the dataset.
- Performed data cleaning (handling missing values, outlier detection).
- Generated summary statistics.
- Created visualizations (histograms, scatter plots, correlation matrix).

### 2. Model Training and Performance Evaluation

**Tools Used:**
- Python
- scikit-learn

**Description:**
Prepared a Python module to train a machine learning model and evaluate its performance. This module is intended for the Machine Learning engineering team at Cognizant.

**Key Steps:**
- Split the data into training and test sets.
- Selected and trained a machine learning model (e.g., linear regression, decision tree).
- Evaluated the model using appropriate metrics (e.g., accuracy, precision, recall, F1 score).
- Output the performance metrics for review.

**Sample Code:**
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load data
data = pd.read_csv('gala_groceries_data.csv')

# Data preprocessing
# ... (include data cleaning steps here)

# Split data
X = data.drop('target', axis=1)
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(report)
```

### 3. Communication of Findings

**Tools Used:**
- Microsoft PowerPoint

**Description:**
Compiled the findings and analysis into a PowerPoint presentation to communicate the results back to the business stakeholders at Gala Groceries.

**Key Steps:**
- Summarized the EDA findings, highlighting key insights and patterns.
- Described the model training process and performance metrics.
- Provided actionable recommendations based on the analysis.
- Created visual aids to support the presentation (charts, graphs, tables).

## Conclusion

The Cognizant Artificial Intelligence Job Simulation on Forage provided a comprehensive experience in data analysis, model training, and business communication. The skills and insights gained from this simulation are directly applicable to real-world AI projects and align with the objectives of Cognizant’s Data Science team.

## Files Included

- `eda_notebook.ipynb`: Jupyter notebook containing the exploratory data analysis.
- `model_training.py`: Python module for model training and performance evaluation.
- `findings_presentation.pptx`: PowerPoint presentation summarizing the findings and recommendations.
  
This README file provides a structured summary of the activities and outcomes of the job simulation, ensuring that all relevant details are clearly documented for future reference.
