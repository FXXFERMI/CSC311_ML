from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import data_cleaning as dc


t_train = dc.t_train
t_test = dc.t_test
t_valid = dc.t_valid

# load dataset
clean_data = pd.read_csv('/Users/fermis/Desktop/CSC311/CSC311_ML/data/pred_data/analysis_dataset.csv')
train_data = pd.read_csv('/Users/fermis/Desktop/CSC311/CSC311_ML/data/pred_data/train_dataset.csv')
valid_data = pd.read_csv('/Users/fermis/Desktop/CSC311/CSC311_ML/data/pred_data/valid_dataset.csv')
test_data = pd.read_csv('/Users/fermis/Desktop/CSC311/CSC311_ML/data/pred_data/test_dataset.csv')

# X_train
features_train_df = train_data.drop(['id', 'Label', 'Q10'], axis=1)
X_train_bow_df = pd.DataFrame(dc.X_train_bow, index=features_train_df.index)
X_train_df = pd.concat([features_train_df, X_train_bow_df], axis=1)
X_train = X_train_df.values
# print(X_train)


# X_valid
features_valid_df = valid_data.drop(['id', 'Label', 'Q10'], axis=1)
X_valid_bow_df = pd.DataFrame(dc.X_valid_bow, index=features_valid_df.index)
X_valid_df = pd.concat([features_valid_df, X_valid_bow_df], axis=1)
X_valid = X_valid_df.values

# X_test
features_test_df = test_data.drop(['id', 'Label', 'Q10'], axis=1)
X_test_bow_df = pd.DataFrame(dc.X_test_bow, index=features_test_df.index)
X_test_df = pd.concat([features_test_df, X_test_bow_df], axis=1)
X_test = X_test_df.values

dtree = DecisionTreeClassifier(random_state=42, max_depth=6)
dtree.fit(X_train, t_train)

train_pre = dtree.predict(X_train)
val_pre = dtree.predict(X_valid)
train_corr = train_pre[train_pre == t_train]
val_corr = val_pre[val_pre == t_valid]
train_acc = len(train_corr) / len(t_train)
val_acc = len(val_corr) / len(t_valid)

print("LR Train Acc:", train_acc)
print("LR Valid Acc:", val_acc)


#### Prediction ####
test_pre = dtree.predict(X_test)
print(f"Test Accuracy: {accuracy_score(t_test, test_pre)}")
print(f"Test Precision: {precision_score(t_test, test_pre, average='macro')}")

#print("Feature importances:")
#for feature, importance in zip(ml.X_train_df.columns, dtree.feature_importances_):
    #print(f"{feature}: {importance}")

# plt.figure(figsize=(20,10))  # 设置图形的大小
# plot_tree(dtree, filled=True, feature_names=X.columns, rounded=True)
# plt.show()


