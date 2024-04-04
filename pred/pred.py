#### Preamble ####
# Introduction: make prediction for all new test set
# Author: Siqi Fei, Runshi Zhang, Mark A Stevens, Adelina Patlatii
# Date: 21 March 2024
# Contact: fermi.fei@mail.utoronto.ca
# License: MIT
# Pre-requisites: install pip pandas, numpy and random

# basic python imports are permitted
import sys
import csv
import random

# numpy and pandas are also permitted
import numpy
import pandas

def predict(x):
    """
    Helper function to make prediction for a given input x.
    This code is here for demonstration purposes only.
    """
    # randomly choose between the four choices: 'Dubai', 'Rio de Janeiro', 'New York City' and 'Paris'.
    # NOTE: make sure to be *very* careful of the spelling/capitalization of the cities!!
    y = random.choice(['Dubai', 'Rio de Janeiro', 'New York City', 'Paris'])

    # return the prediction
    return y


def predict_all(filename):
    """
    Make predictions for the data in filename
    """
    # read the file containing the test data
    # you do not need to use the "csv" package like we are using
    # (e.g. you may use numpy, pandas, etc)
    data = csv.DictReader(open(filename))

    predictions = []
    for test_example in data:
        # obtain a prediction for this test example
        pred = predict(test_example)
        predictions.append(pred)

    return predictions

predictions = predict_all("test_dataset.csv")