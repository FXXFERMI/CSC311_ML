#### Preamble ####
# Introduction: Clean the data, deal with N/A and missing values
# Author: Siqi Fei, Runshi Zhang, Mark A Stevens, Adelina Patlatii
# Date: 21 March 2024
# Contact: fermi.fei@mail.utoronto.ca
# License: MIT
# Pre-requisites: install pip pandas, numpy and random

import sys
import csv
import random

import numpy as np
import pandas as pd

seed = 42
np.random.seed(seed)

#### Import Data ####
# Load the dataset
data_path = 'data/raw_data/clean_dataset.csv'
clean_data = pd.read_csv(data_path)
clean_data_filename = 'data/pred_data/analysis_dataset.csv'

train_data_filename = 'data/pred_data/train_dataset.csv'
valid_data_filename = 'data/pred_data/valid_dataset.csv'
test_data_filename = 'data/pred_data/test_dataset.csv'

#####################################################################################################################
#### Clean Data - EACH Question ####
"""
- We don't have to reformat Q1-4 since they are numerical value
- We have to fill the missing values with median
- There will not have outlier
----------------------------------------------------------------------------------------------------
- We will first clean the Q5 with one-hot encoding, mark the 4 options: Co-worker,Partner,Friends, Siblings
into a one hot vector, like this:

Co-worker,Partner,Friends,Siblings
1,0,0,0

- We will also deal with missing value by replace missing values with 0 if we have more 0; or replace with 1 if 
we have more 1.

- There will not have outlier
----------------------------------------------------------------------------------------------------
- We then clean Q6 follow the same format as above Q5, however, the only change is: since the it has 6 different scale,
we will consider each scale as a individual data, and mark them with their own value:
(1 = 1, 2 = 2, 3 = 3, 4 = 4, 5 = 5, 6 = 6)

look like this:

Skyscrapers,Sport,Art and Music,Carnival,Cuisine,Economic
6,4,2,1,3,5

- We have to fill the missing values with median

- There will not have outlier
----------------------------------------------------------------------------------------------------
- We don't have to reformat Q7-9 since they are numerical value
- For Q7-9, we will replace missing values with median
- For outlier: 
    we first normalize the dataset, find the mean
    * if one data point that fall more than three standard deviations from the mean
    (we used two standard deviations in our case, since three standard deviations will effect the data too far)
    * we will consider the data as an outlier.
    We will replace outlier with mean
----------------------------------------------------------------------------------------------------
- Last, we clean Q10. We will make a list of words in the vocabulary, 
  and produces a data matrix consisting of bag-of-word features, along with a vector of labels.
  (like what we did in lab9)
  
- We will replaced all missing value with 'without'

- There will not have outlier
"""
# Replace empty strings with NaN
clean_data.replace('', np.nan)

#####################################################################################################################
#### Q1 ####
# fill the missing values with median
# Calculate the median of Q1
median_q1 = clean_data['Q1'].median()
# print("Median Q1: ", median_q1)
# Replace missing values in Q1 with its median
clean_data['Q1'] = clean_data['Q1'].fillna(median_q1)

#####################################################################################################################
#### Q2 ####
# fill the missing values with median

# Calculate the median of Q2
median_q2 = clean_data['Q2'].median()

# Replace missing values in Q2 with its median
clean_data['Q2'] = clean_data['Q2'].fillna(median_q2)

#####################################################################################################################
#### Q3 ####
# fill the missing values with median

# Calculate the median of Q3
median_q3 = clean_data['Q3'].median()

# Replace missing values in Q3 with its median
clean_data['Q3'] = clean_data['Q3'].fillna(median_q3)

#####################################################################################################################
#### Q4 ####
# fill the missing values with median

# Calculate the median of Q4
median_q4 = clean_data['Q4'].median()

# Replace missing values in Q4 with its median
clean_data['Q4'] = clean_data['Q4'].fillna(median_q4)

# clean_data.to_csv(clean_data_filename, index=False)

#####################################################################################################################
#### Q5 ####
"""
Find the mode for each option. If the answer has the option, we will add 1 to this option count, if not, add 1 to this 
no_option count. We do same thing for each four options. Then, compare option count and no_option count. 
If no_option count > option count, we will mark the binary outcome of the option as 0, otherwise, it will be marked as 1
Follow above steps, now we have the binary outcome set for the whole dataset, for example: 
{'Friends': 1, 'Siblings': 0, 'Co-worker': 0, 'Partner': 1}
We will then reformat the whole Q5 data into one hot, then replace the missing value with the binary outcome.

"""
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
clean_data = pd.concat([clean_data, dummies], axis=1).drop('Q5', axis=1)
df = pd.concat([clean_data, dummies], axis=1)

#####################################################################################################################
#### Q6 ####
"""
- We first split it to six different variables
- then replace missing values with median
"""
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

# drop Q6
clean_data = clean_data.drop('Q6', axis=1)

#####################################################################################################################
#### Q7 | Q8 | Q9 ####
"""
- We first normalize all the data, then find the outlier,
- after we find all the outliers, we de-normalize all the data
- then replace missing values and outliers with mean
"""
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

    # Identify outliers as values more than 3 standard deviations from the mean
    outliers = (clean_data[column] < -2) | (clean_data[column] > 2)
    # Replace outliers and NaNs with the mean of the column
    clean_data[column] = clean_data[column] * col_std + col_mean
    clean_data[column] = clean_data[column].mask(outliers | clean_data[column].isna(), col_mean)

#####################################################################################################################
#### Q10 ####
"""
- We first cleaned the reviews, make them all lowercase
- removed apostrophes, period, non-standard letter, '\n', '-' etc.
- replace all the missing value with 'without' 
"""


# Function to clean the text in Q10
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


# Covert all to lowercase
clean_data['Q10'] = clean_data['Q10'].str.lower()

# Replace newline, - characters with a space in the 'Q10' column
clean_data['Q10'] = clean_data['Q10'].str.replace('\n', ' ', regex=False)
clean_data['Q10'] = clean_data['Q10'].str.replace('-', ' ', regex=False)

# Apply the clean_text function
clean_data['Q10'] = clean_data['Q10'].apply(clean_text)

# Replace empty strings with 'without'
clean_data['Q10'] = clean_data['Q10'].apply(lambda x: 'without' if x == '' else x)

# save cleaned data
clean_data.to_csv(clean_data_filename, index=False)

# Shuffle the dataset
shuffled_indices = np.random.permutation(clean_data.index)
shuffled_data = clean_data.loc[shuffled_indices]

# Calculate the indices for splitting
total_rows = len(shuffled_data)
train_end = int(total_rows * 0.8)
valid_end = int(total_rows * 0.90)

# Split the data
train_set = shuffled_data.iloc[:train_end]
valid_set = shuffled_data.iloc[train_end:valid_end]
test_set = shuffled_data.iloc[valid_end:]

# Save the splits as CSV files
train_set.to_csv(train_data_filename, index=False)
valid_set.to_csv(valid_data_filename, index=False)
test_set.to_csv(test_data_filename, index=False)

#####################################################################################################################
#####################################################################################################################
#### Generate t_train, t_valid, X_train_bow(only have the BoW of Q10) and X_valid_bow((only have the BoW of Q10) ####
# all the code below References: CSC311 Winter2023-2024 lab9


vocab = list()

for row in train_set['Q10']:
    a = row.lower().split()
    for word in a:
        if word not in vocab:
            vocab.append(word)


# print("Vocabulary Size: ", len(vocab))
# print(vocab)

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
    t = np.zeros([len(data)])
    for i, (review, label) in enumerate(data):
        words = review.split()
        for w in words:
            if w in vocab:
                j = vocab.index(w)
                X[i, j] = 1

        label = label.lower().strip()  # Convert to lowercase and strip whitespace
        if label == "dubai":
            t[i] = 1
        elif label == "rio de janeiro":
            t[i] = 2
        elif label == "new york city":
            t[i] = 3
        elif label == "paris":
            t[i] = 4

    return X, t


data_1 = list(zip(clean_data['Q10'], clean_data['Label']))
data_2 = list(zip(train_set['Q10'], train_set['Label']))
data_3 = list(zip(valid_set['Q10'], valid_set['Label']))
data_4 = list(zip(test_set['Q10'], test_set['Label']))
X_t, t_t = make_bow(data_1, vocab)
X_train_bow, t_train = make_bow(data_2, vocab)
X_valid_bow, t_valid = make_bow(data_3, vocab)
X_test_bow, t_test = make_bow(data_4, vocab)
np.savetxt("data/pred_data/matrix/X_train_bow.csv", X_train_bow,
           delimiter=",", fmt='%i')
np.savetxt("data/pred_data/matrix/t_train.csv", t_train,
           delimiter=",", fmt='%i')
np.savetxt("data/pred_data/matrix/X_valid_bow.csv", X_train_bow,
           delimiter=",", fmt='%i')
np.savetxt("data/pred_data/matrix/t_valid.csv", t_train,
           delimiter=",", fmt='%i')
np.savetxt("data/pred_data/matrix/X_test_bow.csv", X_train_bow,
           delimiter=",", fmt='%i')
np.savetxt("data/pred_data/matrix/t_test.csv", t_train,
           delimiter=",", fmt='%i')

# produce the mapping of words to count - whole dataset
vocab_count_mapping = list(zip(vocab, np.sum(X_t, axis=0)))
vocab_count_mapping = sorted(vocab_count_mapping, key=lambda e: e[1], reverse=True)
# for word, cnt in vocab_count_mapping:
#    print(word, cnt)

# produce the mapping of words to count - train dataset
vocab_2_count_mapping = list(zip(vocab, np.sum(X_train_bow, axis=0)))
vocab_2_count_mapping = sorted(vocab_2_count_mapping, key=lambda e: e[1], reverse=True)
# for word, cnt in vocab_2_count_mapping:
#    print(word, cnt)

# produce the mapping of words to count - validation dataset
vocab_3_count_mapping = list(zip(vocab, np.sum(X_valid_bow, axis=0)))
vocab_3_count_mapping = sorted(vocab_3_count_mapping, key=lambda e: e[1], reverse=True)
# for word, cnt in vocab_3_count_mapping:
#     print(word, cnt)

# produce the mapping of words to count - validation dataset - test dataset
vocab_4_count_mapping = list(zip(vocab, np.sum(X_test_bow, axis=0)))
vocab_4_count_mapping = sorted(vocab_4_count_mapping, key=lambda e: e[1], reverse=True)
# for word, cnt in vocab_3_count_mapping:
#     print(word, cnt)

# print(type(X_train_bow))
