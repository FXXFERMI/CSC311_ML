import pandas as pd
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import precision_score, accuracy_score, mean_squared_error

df = pd.read_csv('/Users/fermis/Desktop/CSC311/CSC311_ML/data/pred_data/train_dataset.csv')
test = pd.read_csv('/Users/fermis/Desktop/CSC311/CSC311_ML/data/pred_data/test_dataset.csv')
valid = pd.read_csv('/Users/fermis/Desktop/CSC311/CSC311_ML/data/pred_data/valid_dataset.csv')

# define keywords
keywords = ['dubai', 'ny', 'new york', 'new york city', 'rio', 'rio de janeiro', 'paris', 'cest la vie',
            'the city of love', 'eiffel', 'apple', 'football', 'soccer', 'rich', 'money', 'burj khalifa']

#### Train dataset ####
# build the train matrix -df now become X_train
for keyword in keywords:
    df[f'keyword_{keyword}'] = df['Q10'].str.count(keyword)

# target
y_train = df['Label']

# clean dataset's list
columns_to_drop = ['Label', 'Q10', 'id']
X_train = df.drop(columns_to_drop, axis=1)  # clean X_train
# print(X_train.head())


#### Model ####
# bild mlp model
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=5000, random_state=42)

# train mlp
mlp.fit(X_train, y_train)

#### Validation ####
# build the valid matrix
for keyword in keywords:
    valid[f'keyword_{keyword}'] = valid['Q10'].str.count(keyword)

# target
y_valid = valid['Label']

# clean dataset's list
columns_to_drop = ['Label', 'Q10', 'id']
X_valid = valid.drop(columns_to_drop, axis=1)  # clean X_valid

train_pre = mlp.predict(X_train)
val_pre = mlp.predict(X_valid)
train_corr = train_pre[train_pre == y_train]
val_corr = val_pre[val_pre == y_valid]
train_acc = len(train_corr) / len(y_train)
val_acc = len(val_corr) / len(y_valid)

print("LR Train Acc:", train_acc)
print("LR Valid Acc:", val_acc)


#### Prediction #####
# X_test
for keyword in keywords:
    test[f'keyword_{keyword}'] = test['Q10'].str.count(keyword)

# target
y_test = test['Label']

# clean X_test
X_test = test.drop(columns_to_drop, axis=1)

# do prediction
y_pred = mlp.predict(X_test)

# compare
accuracy = accuracy_score(y_test, y_pred)
print(f'test Accuracy: {accuracy}')

precision = precision_score(y_test, y_pred, average='macro')

print(f'test Precision: {precision}')
