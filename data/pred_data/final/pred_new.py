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

estimator_path = 'estimators.csv'

## Read the estimator
model_estimator_df = pd.read_csv(estimator_path, header=None)
model_est = model_estimator_df.values


def predict(x):
    """
    Helper function to make prediction for a given input x.
    This code is here for demonstration purposes only.
    """
    # randomly choose between the four choices: 'Dubai', 'Rio de Janeiro', 'New York City' and 'Paris'.
    # NOTE: make sure to be *very* careful of the spelling/capitalization of the cities!!
    y = ['Dubai', 'Rio de Janeiro', 'New York City', 'Paris']

    # Calculate the probability for each city
    probabilities = pred(x, model_est)

    #print("Shape of probabilities:", probabilities)
    # Get the index of the maximum probability
    max_index = np.argmax(probabilities)
    # Predict the city with the highest probability
    prediction = y[max_index]
    #print("Predicted:", prediction)
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
    #####################################################################################################################
    #### Q1 ####
    # We don't have to reformat Q1 since they are numerical value
    # fill the missing values with median
    # No outlier

    # Calculate the median of Q1
    median_q1 = clean_data['Q1'].median()

    # Replace missing values in Q1 with its median
    clean_data['Q1'] = clean_data['Q1'].fillna(median_q1)

    #####################################################################################################################
    #### Q2 ####
    # We don't have to reformat Q2 since they are numerical value
    # fill the missing values with median
    # No outlier

    # Calculate the median of Q2
    median_q2 = clean_data['Q2'].median()

    # Replace missing values in Q2 with its median
    clean_data['Q2'] = clean_data['Q2'].fillna(median_q2)

    #####################################################################################################################
    #### Q3 ####
    # We don't have to reformat Q3 since they are numerical value
    # fill the missing values with median
    # No outlier

    # Calculate the median of Q3
    median_q3 = clean_data['Q3'].median()

    # Replace missing values in Q3 with its median
    clean_data['Q3'] = clean_data['Q3'].fillna(median_q3)

    #####################################################################################################################
    #### Q4 ####
    # We don't have to reformat Q4 since they are numerical value
    # fill the missing values with median
    # No outlier

    median_q4 = clean_data['Q4'].median()

    # Replace missing values in Q4 with its median
    clean_data['Q4'] = clean_data['Q4'].fillna(median_q4)

    #####################################################################################################################
    #### Q5 ####
    # Find the mode for each option. If the answer has the option, we will add 1 to this option count, if not,
    # add 1 to this no_option count. We do same thing for each four options. Then, compare option count and no_option
    # count. If no_option count > option count, we will mark the binary outcome of the option as 0, otherwise,
    # it will be marked as 1 Follow above steps, now we have the binary outcome set for the whole dataset,
    # for example: {'Friends': 1, 'Siblings': 0, 'Co-worker': 0, 'Partner': 1} We will then reformat the whole Q5
    # data into one hot, then replace the missing value with the binary outcome.
    clean_data_Q5 = clean_data['Q5'].str.split(', ')
    for option in ['Partner', 'Friends', 'Siblings', 'Co-worker']:
        clean_data[option] = 0

    for i, options_lst in clean_data_Q5.items():
        #print(i, options_lst)
        if options_lst is not np.nan:
            for lst in options_lst:
                options = lst.split(',')
                for option in options:
                    option = option.strip()  # Remove leading/trailing whitespace
                    if option in ['Partner', 'Friends', 'Siblings', 'Co-worker']:
                        clean_data.at[i, option] = 1

    option_majority = {}
    for option in ['Partner', 'Friends', 'Siblings', 'Co-worker']:
        total_presence = clean_data[option].sum()
        # If more than half of the responses include the option, it's predominantly present
        option_majority[option] = 1 if total_presence >= (len(clean_data) / 2) else 0

    for idx, row in clean_data.iterrows():
        if pd.isnull(clean_data.at[idx, 'Q5']):  # If the original Q5 value is missing
            for option in ['Partner', 'Friends', 'Siblings', 'Co-worker']:
                clean_data.at[idx, option] = option_majority[option]

    # drop Q5 in dataset
    clean_data = clean_data.drop('Q5', axis=1)

    #####################################################################################################################
    #### Q6 ####
    # We first split it to six different variables
    # then replace missing values with median

    clean_data_Q6 = clean_data['Q6'].str.split(',')
    # print(clean_data_Q6.head())
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

    #####################################################################################################################
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

    #####################################################################################################################
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

    #####################################################################################################################
    #### Generate t_train, t_valid, X_train_bow(only have the BoW of Q10) and X_valid_bow((only have the BoW of Q10) ####
    # all the code below References: CSC311 Winter2023-2024 lab9

    vocab_path = 'final_vocab.csv'
    vocab_df = pd.read_csv(vocab_path)
    vocab = vocab_df['word'].tolist()
    #print("Vocabulary Size: ", len(vocab))
    #print(vocab)

    # create X_test_Q10 here
    test_data_lst = list(clean_data['Q10'])
    X_test_bow = make_bow(test_data_lst, vocab)

    # X_test for whole test set
    features_test_df = clean_data.drop(['id', 'Q10'], axis=1)
    X_test_bow_df = pd.DataFrame(X_test_bow, index=features_test_df.index)
    X_test_df = pd.concat([features_test_df, X_test_bow_df], axis=1)
    # print(X_test_df.head())
    predictions = []
    for index, test_example in X_test_df.iterrows():
        #print("Shape of test_example:", test_example.values.shape)
        # obtain a prediction for this test example
        pred_res = predict(test_example.values)
        predictions.append(pred_res)

    return predictions

#####################################################################################################################
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


def extract_category_value(row, category_name):
    for item in row:
        if item.startswith(f'{category_name}=>'):
            value_part = item.split('=>')[1]
            if value_part.isdigit():
                return int(value_part)
    return np.nan  # Return None or a default value if the category is not found


def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / exp_z.sum(axis=0)

def pred(x, w):
    return softmax(np.dot(x, w))

if __name__ == '__main__':
    res = predict_all('final_test_dataset_2.csv')
    print(res)