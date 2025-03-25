# Import Libraries
import pandas as pd
import numpy as np
from datetime import datetime as dt

# Import Visualization Libraries
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Define dataframe having the path to the csv file
df = pd.read_csv("C:\\Users\\guest_1na2lat\\Downloads\\archive.zip")
print("Number of rows are: ",df.shape[0])
print("Number of columns are: ",df.shape[1])
print(df.info())

# Select features wisely to avoid overfitting
# Dropping Unnamed: 32 and id columns for handling missing values
df.drop(["Unnamed: 32", "id"], axis=1, inplace=True)

# vizualizationn
#note: in this project we are taking the column "diagnosis" as the target variable having beneign or malignant values
# This is a pandas function that returns the count of unique values(B/M)in the target column. It sorts the counts in descending order by default.
print(df['diagnosis'].value_counts())

# Mapping 'B' and 'M' to 0 and 1 (optional if you want numerical target labels)
df['diagnosis'] = df['diagnosis'].map({'B': 0, 'M': 1})


# Drop the ' diagnosis' column from features list to make it target variable 
X = df.drop('diagnosis', axis=1)
#assign the values of it to Y as it will become the dependant variable for which the model will predict values
y = df['diagnosis']

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


'''The function will take model, X_train, X_test, y_train, y_test
    and then it will fit the model, then make predictions on the trained model,
    it will print confusion matrix for train and test, and finally return the scores.
    '''
# Function to evaluate model

def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    
# fit the model on the training data - learning from training data

# make predictions on the test and training data
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Classification reports((precision, recall, f1-score))
    report_train = classification_report(y_train, y_pred_train, output_dict=True)
    report_test = classification_report(y_test, y_pred_test, output_dict=True)
    
    # Extract metrics
    precision_train_malignant = report_train['1']['precision']
    precision_train_benign = report_train['0']['precision']
    recall_train_malignant = report_train['1']['recall']
    recall_train_benign = report_train['0']['recall']
    F1_train_malignant = report_train['1']['f1-score']
    F1_train_benign = report_train['0']['f1-score']
    
    precision_test_malignant = report_test['1']['precision']
    precision_test_benign = report_test['0']['precision']
    recall_test_malignant = report_test['1']['recall']
    recall_test_benign = report_test['0']['recall']
    F1_test_malignant = report_test['1']['f1-score']
    F1_test_benign = report_test['0']['f1-score']
    
    return [precision_train_malignant, precision_test_malignant, recall_train_malignant, recall_test_malignant,
            F1_train_malignant, F1_test_malignant, precision_train_benign, precision_test_benign,
            recall_train_benign, recall_test_benign, F1_train_benign, F1_test_benign]
    
#Precision: measures how many of the predicted positive instances are actually positive.
#recall measures how many of the actual positive instances were correctly identified
#f1 score measures harmonic mean of precision and recall
#accuracy measures how many predictions the model got correct out of all the predictions

#initialization: giving the model maximum iterations to train and fit
lr_model = LogisticRegression(fit_intercept=True, max_iter=10000)
#The evaluate_model function will return various performance metrics, which are stored in lr_score. 
#This would t include metrics such as precision, recall, accuracy, and F1 scores for both the training and test data.
lr_score = evaluate_model(lr_model, X_train, X_test, y_train, y_test)

# Create DataFrame to store results
score = pd.DataFrame({
    'Metric': ['Precision Train Malignant', 'Precision Test Malignant', 'Recall Train Malignant', 'Recall Test Malignant',
               'F1 Train Malignant', 'F1 Test Malignant', 'Precision Train Benign', 'Precision Test Benign',
               'Recall Train Benign', 'Recall Test Benign', 'F1 Train Benign', 'F1 Test Benign'],
    'Logistic Regression': lr_score
})

# Print the evaluation metrics
print(score)

from sklearn.ensemble import RandomForestClassifier
# ML Model - 2 Implementation
rf_model = RandomForestClassifier(random_state=0)
rf_score = evaluate_model(rf_model, X_train, X_test, y_train, y_test)
score['Random Forest'] = rf_score
print(score)


import xgboost as xgb
# ML Model - 3 Implementation
xgb_model = xgb.XGBClassifier()
# Visualizing evaluation Metric Score chart
xgb_score = evaluate_model(xgb_model, X_train, X_test, y_train, y_test)
score['XGB'] = xgb_score
print(score)

# computing the number of correct and incorrect predictions across the two classes
# Confusion Matrix
cm_train = confusion_matrix(y_train, rf_model.predict(X_train))
cm_test = confusion_matrix(y_test, rf_model.predict(X_test))

#first subplot for training data and second subplot for testing data 
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
#heatmap for training subplot
sns.heatmap(cm_train, annot=True, fmt="d", cmap="Blues", ax=ax[0])
ax[0].set_title("Train Confusion Matrix")
ax[0].set_xlabel("Predicted")
ax[0].set_ylabel("True")
#heatmap for testing subplot
sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues", ax=ax[1])
ax[1].set_title("Test Confusion Matrix")
ax[1].set_xlabel("Predicted")
ax[1].set_ylabel("True")

#adjusting layout tp prevent overlapping of subplots
plt.tight_layout()
plt.show()

'''Random Forest consistently performs well across all metrics in both training and testing phases, 
making it a strong candidate for the most reliable model in this case.

XGBoost performs similarly to Random Forest but slightly lags in precision and recall on the test data.

Logistic Regression performs reasonably well but has lower recall for malignant tumors and 
slightly lower precision for benign tumors compared to the other two models.
'''
