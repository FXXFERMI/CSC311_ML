import pandas as pd
import os
import numpy as np
from random import Random

raw_data_path = os.path.join('..', '..', 'data', 'raw_data', 'clean_dataset.csv')
analysis_data_path = os.path.join('..', '..', 'data', 'Q5_analysis_data')

# Create the directory if it doesn't exist
os.makedirs(analysis_data_path, exist_ok=True)

clean_data_filename = os.path.join(analysis_data_path, 'Q5_analysis_data.csv')
train_data_filename = os.path.join(analysis_data_path, 'Q5_train_set.csv')
valid_data_filename = os.path.join(analysis_data_path, 'Q5_valid_set.csv')
test_data_filename = os.path.join(analysis_data_path, 'Q5_test_set.csv')

raw_data = pd.read_csv(raw_data_path)

# Replace empty strings with NaN
raw_data.replace('', np.nan, inplace=True)

# Clean the data by removing rows with NA values
cleaned_data = raw_data.dropna()

# Select Q5 specifically and the target column
cleaned_data = cleaned_data[['Q5', 'Label']]

# Split Q5 answers into separate columns and create dummy variables
cleaned_data['Q5'] = cleaned_data['Q5'].str.split(',')
dummies = cleaned_data['Q5'].apply(lambda x: pd.Series(1, index=x)).fillna(0)

# Convert dummy variables to integer
dummies = dummies.astype(int)
cleaned_data = pd.concat([cleaned_data.drop('Q5', axis=1), dummies], axis=1)


# Save cleaned data
cleaned_data.to_csv(clean_data_filename, index=False)

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