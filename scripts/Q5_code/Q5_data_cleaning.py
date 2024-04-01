import pandas as pd
import os
import numpy as np
from random import Random

# cleaned data set path
raw_data_path = os.path.join('..', '..', 'data', 'raw_data', 'clean_dataset.csv')
analysis_data_path = os.path.join('..', '..', 'data', 'Q5_analysis_data')

# train,valid,test dataset paths
clean_data_filename = os.path.join(analysis_data_path, 'Q5_analysis_data.csv')
train_data_filename = os.path.join(analysis_data_path, 'Q5_train_set.csv')
valid_data_filename = os.path.join(analysis_data_path, 'Q5_valid_set.csv')
test_data_filename = os.path.join(analysis_data_path, 'Q5_test_set.csv')

raw_data = pd.read_csv(raw_data_path)

# Replace empty strings with NaN
raw_data.replace('', np.nan, inplace=True)

# Clean the data by removing rows with NA values
cleaned_data = raw_data.dropna()

# Select Q5 specifically
cleaned_data = cleaned_data[['Q5', 'Label']]

# One-Hot
cleaned_data['Q5'] = cleaned_data['Q5'].str.split(',')
dummies = cleaned_data['Q5'].apply(lambda x: pd.Series(1, index=x)).fillna(0)
cleaned_data = pd.concat([cleaned_data, dummies], axis=1).drop('Q5', axis=1)
df = pd.concat([cleaned_data, dummies], axis=1)

# save cleaned data with one-hot
cleaned_data.to_csv(clean_data_filename, index=False)

#### Split Dataset ####
# Shuffle the dataset
r = Random(18)

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

#### Save Data ####
# Save the splits as CSV files
train_set.to_csv(train_data_filename, index=False)
valid_set.to_csv(valid_data_filename, index=False)
test_set.to_csv(test_data_filename, index=False)

