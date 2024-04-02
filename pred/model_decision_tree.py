from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

df = pd.read_csv('/Users/张润石/Desktop/CSC311_ML/data/pred_data/train_dataset.csv')

# define keywords
keywords = ['dubai','ny', 'new york', 'new york city', 'rio','rio de janeiro','paris', 'cest la vie',
            'the city of love', 'eiffel', 'apple', 'football', 'soccer', 'rich', 'money', 'burj khalifa']


# 计算每个关键词在每个标签下的条件概率
keyword_probs = {}
for label in df['Label'].unique():
    label_df = df[df['Label'] == label]
    total_count = len(label_df)
    keyword_probs[label] = {}
    for keyword in keywords:
        # 计算每个关键词在当前标签下的出现次数
        keyword_count = label_df['Q10'].str.contains(keyword).sum()
        # 计算条件概率
        keyword_probs[label][keyword] = keyword_count / total_count

# 为每行数据添加关键词的条件概率作为新特征
for keyword in keywords:
    df[f'prob_{keyword}'] = df.apply(
        lambda row: keyword_probs[row['Label']][keyword] if pd.notnull(row['Q10']) and keyword in row['Q10']
        else 0, axis=1)


y = df['Label']
# 分割特征和标签
columns_to_drop = ['Label', 'Q10','id']
# 仅当关键词确实是列名时才添加到删除列表中
columns_to_drop.extend([keyword for keyword in keywords if keyword in df.columns])

# 删除列
X = df.drop(columns_to_drop, axis=1) # 删除原始的文本列和标签列

dtree = DecisionTreeClassifier(random_state=42, max_depth=3)

dtree.fit(X, y)

print("Feature importances:")
for feature, importance in zip(X.columns, dtree.feature_importances_):
    print(f"{feature}: {importance}")

plt.figure(figsize=(20,10))  # 设置图形的大小
plot_tree(dtree, filled=True, feature_names=X.columns, rounded=True)
plt.show()
