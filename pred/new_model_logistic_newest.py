#### Preamble ####
# Introduction: build logistic regression model, k-fold cross validation
# Author: Siqi Fei, Runshi Zhang, Mark A Stevens, Adelina Patlatii
# Date: 6 April 2024
# Contact: fermi.fei@mail.utoronto.ca
# License: MIT
# Pre-requisites: install pip pandas, numpy, random and sklearn

import sys
import csv
import random
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
import pandas as pd
import data_cleaning as dc
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier

# load dataset
clean_data = pd.read_csv('../data/pred_data/analysis_dataset.csv')
train_data = pd.read_csv('../data/pred_data/train_dataset.csv')
valid_data = pd.read_csv('../data/pred_data/valid_dataset.csv')
test_data = pd.read_csv('../data/pred_data/test_dataset.csv')

# X_train
features_train_df = train_data.drop(['id', 'Label', 'Q10'], axis=1)
X_train_bow_df = pd.DataFrame(dc.X_train_bow, index=features_train_df.index)
X_train_df = pd.concat([features_train_df, X_train_bow_df], axis=1)
X_train = X_train_df.values
# print(X_train)

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


#### Model ####
# References: CSC311 Winter2023-2024 lab9
def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / exp_z.sum(axis=0)


model = LogisticRegression(max_iter=60, solver='lbfgs')
multi_target_logreg = MultiOutputClassifier(model, n_jobs=-1)
multi_target_logreg.fit(X_train, dc.t_train)


###### Prediction ####
train_pre = multi_target_logreg.predict(X_train)
correct_train_predictions = np.all(train_pre == dc.t_train, axis=1)
train_acc = np.mean(correct_train_predictions)
print("LR Train Acc:", train_acc)

val_pre = multi_target_logreg.predict(X_valid)
correct_predictions = np.all(val_pre == dc.t_valid, axis=1)
val_acc = np.mean(correct_predictions)
print("LR Valid Acc:", val_acc)

test_pre = multi_target_logreg.predict(X_test)
correct_predictions = np.all(test_pre == dc.t_test, axis=1)
test_acc = np.mean(correct_predictions)
print("LR test Acc:", test_acc)


##### Store estimator for this model #####
estimators = []
for estimator in multi_target_logreg.estimators_:
    # Append the current estimator to the list
    estimators.append(estimator.coef_)

estimators = np.vstack(estimators).T
# print(coefficients.shape)
coef_path = "../data/pred_data/final/estimators.csv"
np.savetxt(coef_path, estimators, delimiter=',')



#### K-folder cross validation ####
# Define the cross-validator
cv = KFold(n_splits=5, random_state=18, shuffle=True)

scores = cross_val_score(multi_target_logreg, X_train, dc.t_train, cv=cv, scoring='accuracy')

print("Cross-validation scores:", scores)
print("Mean Cross-validation accuracy:", np.mean(scores))