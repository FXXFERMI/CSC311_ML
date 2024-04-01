import pandas as pd
import os
import numpy as np
from random import Random


raw_data_path = os.path.join('..', '..', 'data', 'raw_data', 'clean_dataset.csv')
analysis_data_path = os.path.join('..', '..', 'data', 'Q5_analysis_data')

clean_data_filename = os.path.join(analysis_data_path, 'Q5_analysis_data.csv')
train_data_filename = os.path.join(analysis_data_path, 'Q5_train_set.csv')
valid_data_filename = os.path.join(analysis_data_path, 'Q5_valid_set.csv')
test_data_filename = os.path.join(analysis_data_path, 'Q5_test_set.csv')


raw_data = pd.read_csv(raw_data_path)
raw_data.replace('', np.nan, inplace=True)
cleaned_data = raw_data.dropna(inplace=True)
cleaned_data['Q5'] = cleaned_data['Q5'].str.split(',')
dummies = cleaned_data['Q5'].apply(lambda x: pd.Series(1, index=x)).fillna(0)
cleaned_data = pd.concat([cleaned_data, dummies], axis=1).drop('Q5', axis=1)
df = pd.concat([data, dummies], axis=1)

data.to_csv(clean_data_filename, index=False)

r = Random(118)
shuffled_indices = list(cleaned_data.index)
r.shuffle(shuffled_indices)
shuffled_data = cleaned_data.loc[shuffled_indices]


total_rows = len(shuffled_data)
train_end = int(total_rows * 0.7)
valid_end = int(total_rows * 0.85)


train_set = shuffled_data.iloc[:train_end]
valid_set = shuffled_data.iloc[train_end:valid_end]
test_set = shuffled_data.iloc[valid_end:]

train_set.to_csv(train_data_filename, index=False)
valid_set.to_csv(valid_data_filename, index=False)
test_set.to_csv(test_data_filename, index=False)