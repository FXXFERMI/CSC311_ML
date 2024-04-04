#### Preamble ####
# Introduction: build logistic regression model, k-fold cross validation
# Author: Siqi Fei, Runshi Zhang, Mark A Stevens, Adelina Patlatii
# Date: 21 March 2024
# Contact: fermi.fei@mail.utoronto.ca
# License: MIT
# Pre-requisites: install pip pandas, numpy, random and sklearn

import sys
import csv
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

import data_cleaning as dc

# load dataset
clean_data = pd.read_csv('/Users/fermis/Desktop/CSC311/CSC311_ML/data/pred_data/analysis_dataset.csv')
train_data = pd.read_csv('/Users/fermis/Desktop/CSC311/CSC311_ML/data/pred_data/train_dataset.csv')
valid_data = pd.read_csv('/Users/fermis/Desktop/CSC311/CSC311_ML/data/pred_data/valid_dataset.csv')
test_data = pd.read_csv('/Users/fermis/Desktop/CSC311/CSC311_ML/data/pred_data/test_dataset.csv')

# X_train
features_train_df = train_data.drop(['id', 'Label', 'Q10'], axis=1)
X_train_bow_df = pd.DataFrame(dc.X_train_bow, index=features_train_df.index)
X_train_df = pd.concat([features_train_df, X_train_bow_df], axis=1)
X_train = X_train_df.values
# print(X_train)

# X_train_for final
#features_train_df = clean_data.drop(['id', 'Label', 'Q10'], axis=1)
#X_t_df = pd.DataFrame(dc.X_t, index=features_train_df.index)
#X_t_df = pd.concat([features_train_df, X_t_df], axis=1)
#X_train = X_t_df.values

# X_valid
features_valid_df = valid_data.drop(['id', 'Label', 'Q10'], axis=1)
X_valid_bow_df = pd.DataFrame(dc.X_valid_bow, index=features_valid_df.index)
X_valid_df = pd.concat([features_valid_df, X_valid_bow_df], axis=1)
X_valid = X_valid_df.values

# X_test
features_test_df = test_data.drop(['id', 'Label', 'Q10'], axis=1)
X_test_bow_df = pd.DataFrame(dc.X_test_bow, index=features_test_df.index)
X_test_df = pd.concat([features_test_df, X_test_bow_df], axis=1)
X_test = X_test_df.values

# Using pandas to get descriptive statistics
#print(X_train_bow_df.describe())
#np.savetxt("/Users/fermis/Desktop/CSC311/CSC311_ML/data/pred_data/matrix/X_train.csv", X_train, delimiter=",", fmt='%i')

# Save to CSV
#train_final_dataset_path = '/Users/fermis/Desktop/CSC311/CSC311_ML/data/pred_data/train_final_dataset.csv'
## Save coefficients to their own CSV file
#X_train_df.to_csv(train_final_dataset_path, index=False)


##### K-fold cross-validation ####
def create_k_folds(data, k=5):
    data_shuffled = data.sample(frac=1).reset_index(drop=True)  # Shuffle your dataset
    folds = np.array_split(data_shuffled, k)  # Split dataset into k folds
    return folds


k_folds = create_k_folds(clean_data, k=5)


def train_and_evaluate_k_fold(folds, model_init_func):
    acc_scores = []  # Store accuracy scores for each fold

    for i in range(len(folds)):
        # Prepare training and validation data
        train_folds = [folds[j] for j in range(len(folds)) if j != i]
        train_data = pd.concat(train_folds)
        valid_data = folds[i]

        # Prepare features and labels
        X_train = train_data.drop(['id', 'Label', 'Q10'], axis=1).values
        y_train = train_data['Label'].values
        X_valid = valid_data.drop(['id', 'Label', 'Q10'], axis=1).values
        y_valid = valid_data['Label'].values

        # Initialize and train the model
        model = model_init_func()
        model.fit(X_train, y_train)

        # Predict and evaluate
        predictions = model.predict(X_valid)
        accuracy = np.mean(predictions == y_valid)
        acc_scores.append(accuracy)

        print(f"Fold {i + 1}: Accuracy = {accuracy}")

    return np.mean(acc_scores), acc_scores  # Return the mean accuracy across all folds and scores per fold


# References: CSC311 Winter2023-2024 lab9
# Logistic Regression Model
model = LogisticRegression(max_iter=2000)
model.fit(X_train, dc.t_t)
train_pre = model.predict(X_train)
val_pre = model.predict(X_valid)
train_corr = train_pre[train_pre == dc.t_t]
val_corr = val_pre[val_pre == dc.t_valid]
train_acc = len(train_corr) / len(dc.t_t)
val_acc = len(val_corr) / len(dc.t_valid)

print("LR Train Acc:", train_acc)
print("LR Valid Acc:", val_acc)


# Example model initialization function for Logistic Regression
def init_log_reg_model():
    return LogisticRegression(max_iter=1000)


# DataFrame clean_data with 'Label' as the target column
mean_acc, scores = train_and_evaluate_k_fold(k_folds, init_log_reg_model)
print(f"Mean Accuracy: {mean_acc}")

##### Prediction ####
test_pred = model.predict(X_test)
print(f"Test Accuracy: {accuracy_score(dc.t_test, test_pred)}")
print(f"Precision:{precision_score(dc.t_test, test_pred, average='macro')}")

#### Store coefficients and intercept for this model #####
print("Model intercept:", model.intercept_)
intercept = model.intercept_
# Save intercept to its own text file
intercept_path = '/Users/fermis/Desktop/CSC311/CSC311_ML/data/pred_data/intercept.txt'
with open(intercept_path, 'w') as file:
    file.write(f"Model intercept: {intercept}\n")

## for final inter
#intercept = model.intercept_
## Save intercept to its own text file
#intercept_path = '/Users/fermis/Desktop/CSC311/CSC311_ML/data/pred_data/final/final_intercept.txt'
#with open(intercept_path, 'w') as file:
#    file.write(f"Model intercept: {intercept}\n")

coefficients = []
#print("Model coefficients:")
for feature, cof in zip(X_train_df.columns, model.coef_[0]):
    #print(f"{feature}: {cof}")
    coefficients.append(cof)

features = X_train_df.columns
# Combine the feature names and coefficients into a DataFrame
coefs_df = pd.DataFrame({
    'Feature': features,
    'Coefficient': coefficients
})

# for final coef
#coefficients = []
##print("Model coefficients:")
#for feature, cof in zip(X_t_df.columns, model.coef_[0]):
#    #print(f"{feature}: {cof}")
#    coefficients.append(cof)
#
#features = X_t_df.columns
## Combine the feature names and coefficients into a DataFrame
#coefs_df = pd.DataFrame({
#   'Feature': features,
#   'Coefficient': coefficients
#})
# Save to CSV
csv_file_path = '/Users/fermis/Desktop/CSC311/CSC311_ML/data/pred_data/coefficients.csv'
# Save coefficients to their own CSV file
coefs_df.to_csv(csv_file_path, index=False)

# Save to CSV
#sv_file_path = '/Users/fermis/Desktop/CSC311/CSC311_ML/data/pred_data/final/final_coefficients.csv'
# Save coefficients to their own CSV file
#oefs_df.to_csv(csv_file_path, index=False)