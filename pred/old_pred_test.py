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
import numpy as np
import pandas as pd

intercept_path = '/Users/fermis/Desktop/CSC311/CSC311_ML/data/pred_data/final/final_intercept.txt'
coefficients_path = '/Users/fermis/Desktop/CSC311/CSC311_ML/data/pred_data/final/final_coefficients.csv'
train_path = '/Users/fermis/Desktop/CSC311/CSC311_ML/data/pred_data/final/final_train_dataset.csv'
train_data = pd.read_csv(train_path)

# Read the intercept value from the file
with open(intercept_path, 'r') as file:
    line = file.readline()
    intercept_str = line.strip().split(": ")[1]
    #print(intercept_str)
    numbers_str = intercept_str.strip('[]').split()
    model_intercept = [float(num) for num in numbers_str]

# Read the coefficients into a DataFrame and then extract the coefficients column
model_coefs_df = pd.read_csv(coefficients_path)
model_coefs = model_coefs_df.iloc[:, 1:].values

def predict(x):
    """
    Helper function to make prediction for a given input x.
    This code is here for demonstration purposes only.
    """
    # randomly choose between the four choices: 'Dubai', 'Rio de Janeiro', 'New York City' and 'Paris'.
    # NOTE: make sure to be *very* careful of the spelling/capitalization of the cities!!
    y = ['Dubai', 'Rio de Janeiro', 'New York City', 'Paris']

    # calculate the probability of each city name
    probabilities = []
    # Calculate the probability for each city
    for inter in model_intercept:
        linear_combination = np.dot(x, model_coefs) + inter
        probability = 1 / (1 + np.exp(-linear_combination))  # Sigmoid function
        probabilities.append(probability)

    # Get the index of the maximum probability
    max_index = np.argmax(probabilities)
    # Predict the city with the highest probability
    prediction = y[max_index]
    return prediction


def predict_all(filename):
    """
    Make predictions for the data in filename
    """
    # read the file containing the test data
    # you do not need to use the "csv" package like we are using
    # (e.g. you may use numpy, pandas, etc.)

    #### Load Data ####
    clean_data = pd.read_csv(filename)

    #### Data Clean ####
    # Replace empty strings with NaN for future use
    # We can not simply drop everything because it will affect our matrix size
    clean_data.replace('', np.nan)

    #### Q1 ####
    # We don't have to reformat Q1 since they are numerical value
    # fill the missing values with median
    # No outlier

    # Calculate the median of Q1
    median_q1 = clean_data['Q1'].median()

    # Replace missing values in Q1 with its median
    clean_data['Q1'] = clean_data['Q1'].fillna(median_q1)

    #### Q2 ####
    # We don't have to reformat Q2 since they are numerical value
    # fill the missing values with median
    # No outlier

    # Calculate the median of Q2
    median_q2 = clean_data['Q2'].median()

    # Replace missing values in Q2 with its median
    clean_data['Q2'] = clean_data['Q2'].fillna(median_q2)

    #### Q3 ####
    # We don't have to reformat Q3 since they are numerical value
    # fill the missing values with median
    # No outlier

    # Calculate the median of Q3
    median_q3 = clean_data['Q3'].median()

    # Replace missing values in Q3 with its median
    clean_data['Q3'] = clean_data['Q3'].fillna(median_q3)

    #### Q4 ####
    # We don't have to reformat Q4 since they are numerical value
    # fill the missing values with median
    # No outlier

    median_q4 = clean_data['Q4'].median()

    # Replace missing values in Q4 with its median
    clean_data['Q4'] = clean_data['Q4'].fillna(median_q4)

    #### Q5 ####
    # Find the mode for each option. If the answer has the option, we will add 1 to this option count, if not,
    # add 1 to this no_option count. We do same thing for each four options. Then, compare option count and no_option
    # count. If no_option count > option count, we will mark the binary outcome of the option as 0, otherwise,
    # it will be marked as 1 Follow above steps, now we have the binary outcome set for the whole dataset,
    # for example: {'Friends': 1, 'Siblings': 0, 'Co-worker': 0, 'Partner': 1} We will then reformat the whole Q5
    # data into one hot, then replace the missing value with the binary outcome.

    clean_data_Q5 = clean_data['Q5'].str.split(', ')

    # Flatten the list to get all unique options
    all_individual_options = set(
        [item for sublist in clean_data_Q5.dropna().tolist() for combined in sublist for item in combined.split(',')])

    # Initialize counters for presence of each option
    option_presence_counts = {option.strip(): 0 for option in all_individual_options}

    # Count the presence of each option in each list
    for sublist in clean_data_Q5.dropna().tolist():
        for combined in sublist:
            for item in combined.split(','):
                option_presence_counts[item.strip()] += 1

    total_lists = len(clean_data_Q5.dropna())

    # Determine the binary outcome for each option (1 for presence, 0 for absence)
    binary_outcomes = {}
    for option, count in option_presence_counts.items():
        # If the option is present in more than half of the lists, it's considered as 1 (presence)
        binary_outcomes[option] = 1 if count > total_lists / 2 else 0

    # print(binary_outcomes)

    # find the replacement of NaN value in Dataset Q5, and replace it with replacement
    replacement = []
    for option, res in binary_outcomes.items():
        if res == 1 or res == 1.0:
            replacement.append(option)
    # print(replacement)

    clean_data['Q5'] = clean_data['Q5'].str.split(',')
    # print(clean_data['Q5'][120:129])
    clean_data['Q5'] = clean_data['Q5'].apply(lambda x: x if isinstance(x, list) else replacement)
    # print(clean_data['Q5'][120:129])

    # one-hot
    dummies = clean_data['Q5'].apply(lambda x: pd.Series(1, index=x)).fillna(0)

    # drop Q5
    clean_data = pd.concat([clean_data, dummies], axis=1).drop('Q5', axis=1)

    #### Q6 ####
    # We first split it to six different variables
    # then replace missing values with median

    clean_data_Q6 = clean_data['Q6'].str.split(',')

    # print(clean_data_Q6.head())

    def extract_category_value(row, category_name):
        for item in row:
            if item.startswith(f'{category_name}=>'):
                value_part = item.split('=>')[1]
                if value_part.isdigit():
                    return int(value_part)
        return np.nan  # Return None or a default value if the category is not found

    categories = ['Skyscrapers', 'Sport', 'Art and Music', 'Carnival', 'Cuisine', 'Economic']

    for category in categories:
        # Apply the extractor function for each category
        clean_data[category] = clean_data_Q6.apply(lambda x: extract_category_value(x, category))

    # Find median of each category and replace missing values
    for category in categories:
        # Calculate the median for the current category
        median_value = clean_data[category].median()
        # Replace missing values in the current category with its median
        clean_data[category] = clean_data[category].fillna(median_value)
        # print(f"The median of {category} is: {median_value} and missing values have been replaced.")

    # drop Q6 in dataset
    clean_data = clean_data.drop('Q6', axis=1)

    #### Q7 | Q8 | Q9 ####
    # We first normalize all the data, then find the outlier,
    # after we find all the outliers, we de-normalize all the data
    # then replace missing values and outliers with mean

    for column in ['Q7', 'Q8', 'Q9']:
        clean_data[column] = pd.to_numeric(clean_data[column], errors='coerce')

    # Now, calculate the mean and standard deviation for Q7 to Q9.
    for column in ['Q7', 'Q8', 'Q9']:
        col_mean = clean_data[column].mean()
        col_std = clean_data[column].std()
        # print(col_mean)
        # print(col_std)
        # Normalize the column
        clean_data[column] = (clean_data[column] - col_mean) / col_std

        # Identify outliers as values more than 2 standard deviations from the mean
        outliers = (clean_data[column] < -2) | (clean_data[column] > 2)
        # Replace outliers and NaNs with the mean of the column
        clean_data[column] = clean_data[column] * col_std + col_mean
        clean_data[column] = clean_data[column].mask(outliers | clean_data[column].isna(), col_mean)

    #### Q10 ####
    # We first cleaned the reviews, make them all lowercase
    # removed apostrophes, period, non-standard letter, '\n', '-' etc.
    # replace all the missing value with 'without'

    # Covert all to lowercase
    clean_data['Q10'] = clean_data['Q10'].str.lower()

    # Replace newline, - characters with a space in the 'Q10' column
    clean_data['Q10'] = clean_data['Q10'].str.replace('\n', ' ', regex=False)
    clean_data['Q10'] = clean_data['Q10'].str.replace('-', ' ', regex=False)

    # Apply the clean_text function
    clean_data['Q10'] = clean_data['Q10'].apply(clean_text)

    # Replace empty strings with 'without'
    clean_data['Q10'] = clean_data['Q10'].apply(lambda x: 'without' if x == '' else x)

    #### Generate t_train, t_valid, X_train_bow(only have the BoW of Q10) and X_valid_bow((only have the BoW of Q10) ####
    # all the code below References: CSC311 Winter2023-2024 lab9

    vocab_path = '/Users/fermis/Desktop/CSC311/CSC311_ML/data/pred_data/final/final_vocab.csv'
    vocab_df = pd.read_csv(vocab_path)
    vocab = vocab_df['word'].tolist()
    #print("Vocabulary Size: ", len(vocab))
    #print(vocab)

    # create X_test_Q10 here
    test_data_lst = list(clean_data['Q10'])
    X_test_bow = make_bow(test_data_lst, vocab)

    # X_test for whole test set
    features_test_df = clean_data.drop(['id', 'Q10'], axis=1, errors='ignore')
    X_test_bow_df = pd.DataFrame(X_test_bow, index=features_test_df.index)
    X_test_df = pd.concat([features_test_df, X_test_bow_df], axis=1)
    X_test_df = X_test_df[1:]
    #print(X_test_df.head())
    predictions = []
    for index, test_example in X_test_df.iterrows():
        # obtain a prediction for this test example
        pred = predict(test_example.values)
        predictions.append(pred)

    return predictions


######## Helper Functions #######
# helper function to clean the text in Q10
def clean_text(text):
    if isinstance(text, str):
        # Replace apostrophes with nothing
        text = text.replace("'", "")
        # Keep only standard letters (A-Z, a-z), numbers, commas, and spaces
        cleaned_text = ''.join([char for char in text if char.isalnum() or char in [' '] and char.isascii()])
        return cleaned_text
    else:
        # If 'text' is not a string (e.g., NaN), return 'without'
        return 'without'


def make_bow(data, vocab):
    """
    Produce the bag-of-word representation of the data, along with a vector
    of labels. You *may* use loops to iterate over `data`. However, your code
    should not take more than O(len(data) * len(vocab)) to run.

    Parameters:
        `data`: a list of `(review, label)` pairs, like those produced from
                `list(csv.reader(open("trainvalid.csv")))`
        `vocab`: a list consisting of all unique words in the vocabulary

    Returns:
        `X`: A data matrix of bag-of-word features. This data matrix should be
             a numpy array with shape [len(data), len(vocab)].
             Moreover, `X[i,j] == 1` if the review in `data[i]` contains the
             word `vocab[j]`, and `X[i,j] == 0` otherwise.
        `t`: A numpy array of shape [len(data)], with `t[i] == 1` if
             `data[i]` is a positive review, and `t[i] == 0` otherwise.
    """
    X = np.zeros([len(data), len(vocab)])
    for i, review in enumerate(data):
        words = review.split()
        for w in words:
            if w in vocab:
                j = vocab.index(w)
                X[i, j] = 1

    return X


if __name__ == '__main__':
    res = predict_all('/Users/fermis/Desktop/CSC311/CSC311_ML/data/pred_data/final/final_test_dataset_1.csv')
    print(res)