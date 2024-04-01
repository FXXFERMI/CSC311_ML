import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# Load the dataset
data_path = '~/PycharmProjects/CSC311_ML/data/raw_data/clean_dataset.csv'  # Make sure to replace this with the actual path
data = pd.read_csv(data_path)

# Selecting only the first four Q columns and the Label
features = data[['Q1', 'Q2', 'Q3', 'Q4']]
target = data['Label']

# Splitting the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
# First, we will use `train_test_split` to split the data set into
# 6500 training+validation, and 1500 test:
X_tv, X_test, t_tv, t_test = train_test_split(features, target, test_size=1 / 9,
                                              random_state=1)

# Then, use `train_test_split` to split the training+validation data
# into 5000 train and 1500 validation
X_train, X_valid, t_train, t_valid = train_test_split(X_tv, t_tv,
                                                      test_size=1 / 8,
                                                      random_state=1)

# Handling missing values by replacing them with the mean of their respective columns
X_train_filled = X_train.fillna(X_train.mean())
X_test_filled = X_test.fillna(X_train.mean())

# Standardizing the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_filled)
X_test_scaled = scaler.transform(X_test_filled)

"""
# Logistic Regression
log_reg = LogisticRegression(solver='newton-cg', penalty='l2', C=0.001, max_iter=1000)
log_reg.fit(X_train_scaled, t_train)
y_pred_log_reg = log_reg.predict(X_test_scaled)
accuracy_log_reg = accuracy_score(t_test, y_pred_log_reg)"""

# MLP Classifier
"""max_accuracy = 0
i_max = 0
j_max = 0
for i in range(87, 88):
    for j in range(80, 90):
        mlp = MLPClassifier((i,j), max_iter=2000)
        mlp.fit(X_train_scaled, t_train)
        y_pred_mlp = mlp.predict(X_test_scaled)
        accuracy_mlp = accuracy_score(t_test, y_pred_mlp)
        max_accuracy = max(max_accuracy, accuracy_mlp)
        if accuracy_mlp == max_accuracy:
            i_max = i
            j_max = j
        print("Max accuracy is: " + str(
            max_accuracy) + "    and the values of i and j are:" + str(
            i_max) + "   " + str(j_max))"""
mlp = MLPClassifier((83,), max_iter=2000)
mlp.fit(X_train_scaled, t_train)
y_pred_mlp = mlp.predict(X_test_scaled)
accuracy_mlp = accuracy_score(t_test, y_pred_mlp)
# decision tree
"""
def build_all_decision_trees(max_depths,
                     min_samples_split,
                     criterion,
                     X_train=X_train,
                     t_train=t_train,
                     X_valid=X_valid,
                     t_valid=t_valid):
    out = {}

    for d in max_depths:
        for s in min_samples_split:
            out[(d, s)] = {}
            # Create a DecisionTreeClassifier based on the given hyperparameters and fit it to the data
            tree = DecisionTreeClassifier(criterion=criterion, max_depth=d,
                                          min_samples_split=s)
            tree = tree.fit(X_train, t_train)

            out[(d, s)]['val'] = tree.score(X_valid, t_valid)
            out[(d, s)]['train'] = tree.score(X_train, t_train)
    return out

# Hyperparameters values to try in our grid search
criterions = ["entropy", "gini"]
max_depths = [1, 5, 10, 15, 20, 25, 30, 50, 100]
min_samples_split = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

for criterion in criterions:
    print("\nUsing criterion {}".format(criterion))

    res = build_all_decision_trees(max_depths, min_samples_split, criterion,
                           X_train=X_train, t_train=t_train, X_valid=X_valid,
                           t_valid=t_valid)

    best_val_accuracy = 0
    best_train_accuracy = 0
    best_parameters = [0, 0]
    for d, s in res:
        current_accuracy = res[(d, s)]['val']
        if current_accuracy > best_val_accuracy:
            best_val_accuracy = current_accuracy
            best_train_accuracy = res[(d, s)]['train']
            best_parameters[0] = d
            best_parameters[1] = s

    print("Max_depth = " + str(
        best_parameters[0]) + "; Min_samples_split = " + str(
        best_parameters[1]) + "; Training_accuracy:" + str(
        best_train_accuracy) + ";  Validation_accuracy:" + str(
        best_val_accuracy))






decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train_filled, t_train)  # Using filled data without scaling
y_pred_decision_tree = decision_tree.predict(X_test_filled)
accuracy_decision_tree = accuracy_score(t_test, y_pred_decision_tree)"""

# Print the accuracy scores
# print(f"Logistic Regression Accuracy: {accuracy_log_reg}")
print(f"MLP Classifier Accuracy: {accuracy_mlp}")
# print(f"Decision Tree Classifier Accuracy: {accuracy_decision_tree}")
