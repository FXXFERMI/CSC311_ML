#### Preamble ####
# Purpose:
# Author: Siqi Fei
# Date: 21 March 2024
# Contact: fermi.fei@mail.utoronto.ca
# License: MIT
# Pre-requisites:

import pandas as pd
import os
import numpy as np
from random import Random


#### Data Path ####
raw_data_path = os.path.join('..', '..', 'data', 'raw_data', 'clean_dataset.csv')
analysis_data_path = os.path.join('..', '..', 'data', 'Q10_analysis_data')

# Define file names for the datasets
clean_data_filename = os.path.join(analysis_data_path, 'analysis_data.csv')
train_data_filename = os.path.join(analysis_data_path, 'train_set.csv')
valid_data_filename = os.path.join(analysis_data_path, 'valid_set.csv')
test_data_filename = os.path.join(analysis_data_path, 'test_set.csv')

### Clean Data ####
# Read the CSV file
raw_data = pd.read_csv(raw_data_path)

# Replace empty strings with NaN
raw_data.replace('', np.nan, inplace=True)
# Clean the data by removing rows with NA values
cleaned_data = raw_data.dropna()

# Select Q10 specifically
cleaned_data = cleaned_data[['Q10', 'Label']]

# save cleaned data
cleaned_data.to_csv(clean_data_filename, index=False)

#### Split Dataset ####
# Shuffle the dataset
r = Random(118)
shuffled_indices = list(cleaned_data.index)
r.shuffle(shuffled_indices)
shuffled_data = cleaned_data.loc[shuffled_indices]

# Calculate the indices for splitting
total_rows = len(shuffled_data)
train_end = int(total_rows * 0.7)
valid_end = int(total_rows * 0.85)

# Split the data
train_set = shuffled_data.iloc[:train_end]
valid_set = shuffled_data.iloc[train_end:valid_end]
test_set = shuffled_data.iloc[valid_end:]

# Save the splits as CSV files
train_set.to_csv(train_data_filename, index=False)
valid_set.to_csv(valid_data_filename, index=False)
test_set.to_csv(test_data_filename, index=False)

