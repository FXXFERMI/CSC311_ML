from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

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

dtree = DecisionTreeClassifier(random_state=42, max_depth=3)

dtree.fit(X, y)

print("Feature importances:")
for feature, importance in zip(X.columns, dtree.feature_importances_):
    print(f"{feature}: {importance}")

plt.figure(figsize=(20,10))  # adjust plot size
plot_tree(dtree, filled=True, feature_names=X.columns, rounded=True)
plt.show()
