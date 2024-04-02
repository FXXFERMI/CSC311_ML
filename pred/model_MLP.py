import pandas as pd
from sklearn.neural_network import MLPClassifier


df = pd.read_csv('/Users/张润石/Desktop/CSC311_ML/data/pred_data/train_dataset.csv')

# define keywords
keywords = ['dubai','ny', 'new york', 'new york city', 'rio','rio de janeiro','paris', 'cest la vie',
            'the city of love', 'eiffel', 'apple', 'football', 'soccer', 'rich', 'money', 'burj khalifa']


# calculate the conditional probability of each keyword under each label
keyword_probs = {}
for label in df['Label'].unique():
    label_df = df[df['Label'] == label]
    total_count = len(label_df)
    keyword_probs[label] = {}
    for keyword in keywords:
        # calculate times the keywords appeared under specific label
        keyword_count = label_df['Q10'].str.contains(keyword).sum()
        # calculate the presence of keywords
        keyword_probs[label][keyword] = keyword_count / total_count

# add probability keywords as new feature
for keyword in keywords:
    df[f'prob_{keyword}'] = df.apply(
        lambda row: keyword_probs[row['Label']][keyword] if pd.notnull(row['Q10']) and keyword in row['Q10']
        else 0, axis=1)


y = df['Label']
# split label and features
columns_to_drop = ['Label', 'Q10','id']
# add to drop list when the keywords really in the col names
columns_to_drop.extend([keyword for keyword in keywords if keyword in df.columns])

# delete the col
X = df.drop(columns_to_drop, axis=1) # delete origin useless cols

# build mlp model
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)


# train model
mlp.fit(X, y)

for i, (weights, biases) in enumerate(zip(mlp.coefs_, mlp.intercepts_)):
    print(f"Weights of layer {i}:")
    print(weights)
    print(f"Biases of layer {i}:")
    print(biases)