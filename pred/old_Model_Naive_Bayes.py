from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, precision_score
import pandas as pd
from sklearn.linear_model import LogisticRegression
import data_cleaning as dc

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



##### K-fold cross-validation ####
def create_k_folds(data, k=5):
    data_shuffled = data.sample(frac=1).reset_index(drop=True)  # Shuffle your dataset
    folds = np.array_split(data_shuffled, k)  # Split dataset into k folds
    return folds


k_folds = create_k_folds(clean_data, k=5)


def train_and_evaluate_k_fold(folds, model_init_func):
    acc_scores = []  # Store accuracy scores for each fold

    for i in range(len(folds)):
        # Prepare training and validation data
        train_folds = [folds[j] for j in range(len(folds)) if j != i]
        train_data = pd.concat(train_folds)
        valid_data = folds[i]

        # Prepare features and labels
        X_train = train_data.drop(['id', 'Label', 'Q10'], axis=1).values
        y_train = train_data['Label'].values
        X_valid = valid_data.drop(['id', 'Label', 'Q10'], axis=1).values
        y_valid = valid_data['Label'].values

        # Initialize and train the model
        model = model_init_func()
        model.fit(X_train, y_train)

        # Predict and evaluate
        predictions = model.predict(X_valid)
        accuracy = np.mean(predictions == y_valid)
        acc_scores.append(accuracy)

        print(f"Fold {i + 1}: Accuracy = {accuracy}")

    return np.mean(acc_scores), acc_scores  # Return the mean accuracy across all folds and scores per fold


def init_gaussian_naive_bayes_model():
    return GaussianNB()

# Replace MultinomialNB with GaussianNB for K-fold cross validation
mean_acc, scores = train_and_evaluate_k_fold(k_folds, init_gaussian_naive_bayes_model)
print(f"Mean Accuracy with Gaussian Naive Bayes: {mean_acc}")

# Training and evaluating Gaussian Naive Bayes model
gnb_model = GaussianNB()
gnb_model.fit(X_train, dc.t_train)  # Train the model on the training data
train_pre = gnb_model.predict(X_train)  # Predict on training data
val_pre = gnb_model.predict(X_valid)  # Predict on validation data

# Calculate accuracies
train_acc = np.mean(train_pre == dc.t_train)
val_acc = np.mean(val_pre == dc.t_valid)

print("GNB Train Acc:", train_acc)
print("GNB Valid Acc:", val_acc)

# Prediction on test data
test_pred = gnb_model.predict(X_test)
print(f"Test Accuracy with Gaussian Naive Bayes: {accuracy_score(dc.t_test, test_pred)}")
print(f"Precision with Gaussian Naive Bayes: {precision_score(dc.t_test, test_pred, average='macro')}")