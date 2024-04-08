from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import data_cleaning as dc
import numpy as np
import seaborn as sns

seed = 42
np.random.seed(seed)

class Expert:
    def __init__(self, models, expert):
        self.models = models
        self.expert = expert
        self.classes = 0

    def first_predictions(self, X):
        """
        Returns the prediction for every datapoint for every model.
        size is (X.shape[0], # models)
        """
        predictions = np.zeros((X.shape[0], len(self.models)))
        for i, model in enumerate(self.models):
            predictions[:, i] = model.predict(X)
        # print(predictions)
        return predictions

    def setup(self, X, t):
        self.predictions = self.first_predictions(X)
        self.classes = np.unique(t)

    def vote(self, X):
        """
        Return the prediction on a dataset X using a voting method.
        Very simple, just looks at the majority answer for a given datapoint.
        """
        predictions = self.first_predictions(X)
        final_prediction = np.array([np.bincount(row).argmax() for row in predictions.astype('int64')])
        # print(final_prediction)
        return final_prediction

    def score(self, pred, t):
        # pred = np.zeros((t.shape[0],)).astype('int64')
        t = t.astype('int64')
        # print(pred)
        # print(np.count_nonzero(pred == t))
        correct = np.count_nonzero(pred == t)
        accuracy = correct / t.shape[0]
        return accuracy

    def expert_predict(self, X):
        """
        Turns the original very many feature input
        into the prediction of each expert.
        Then, performs the prediction!
        """
        predictions = self.first_predictions(X)
        return self.expert.predict(predictions)

    def expert_score(self, X, t):
        """
        Transforms the original input into the predictions from experts.
        Then, scores the expert based on this new input.
        """
        predictions = self.first_predictions(X)
        score = self.expert.score(predictions, t)
        return score

    def fit(self, X = None, t=None):
        if X == None:
            predictions = self.predictions
        else:
            predictions = self.first_predictions(X)
        self.expert.fit(predictions, t)

    def partial_fit(self, X = None, t=None, n_iter=100):
        """
        Partially fits the model based on the number of iterations
        """
        if X == None:
            predictions = self.predictions
        else:
            predictions = self.first_predictions(X)

        for i in range(n_iter):
            self.expert.partial_fit(predictions, t, classes=self.classes)







if __name__ == '__main__':

    t_train = dc.t_train
    t_test = dc.t_test
    t_valid = dc.t_valid

    # load dataset
    clean_data = pd.read_csv('data/pred_data/analysis_dataset.csv')
    train_data = pd.read_csv('data/pred_data/train_dataset.csv')
    valid_data = pd.read_csv('data/pred_data/valid_dataset.csv')
    test_data = pd.read_csv('data/pred_data/test_dataset.csv')

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

    # -- Best Validation Models! --

    dtree = DecisionTreeClassifier(max_depth=3)
    dtree.fit(X_train, t_train)

    mlp = MLPClassifier(max_iter=100, hidden_layer_sizes=(7, 7), activation='relu')
    mlp.fit(X_train, t_train)

    lr = LogisticRegression(max_iter=100, C=2)
    lr.fit(X_train, t_train)

    r = 6

    print(f'-- Decision Tree --')
    print(f'accuracy: {round(dtree.score(X_train, t_train), r)}')
    print(f'validation: {round(dtree.score(X_valid, t_valid), r)}')

    print()
    print(f'-- MLP --')
    print(f'accuracy: {round(mlp.score(X_train, t_train), r)}')
    print(f'validation: {round(mlp.score(X_valid, t_valid), r)}')

    print()
    print(f'-- Logistic Regression --')
    print(f'accuracy: {round(lr.score(X_train, t_train), r)}')
    print(f'validation: {round(lr.score(X_valid, t_valid), r)}')









    models = [mlp, lr, dtree]
    expert = MLPClassifier(max_iter=100, hidden_layer_sizes=(7, 7), activation='relu')
    E1 = Expert(models, expert)









    train_acc = E1.score(E1.vote(X_train), t_train)
    valid_acc = E1.score(E1.vote(X_valid), t_valid)

    print()
    print(f'Ensemble: ')
    print(f'Training Accuracy: {round(train_acc, r)}')
    print(f'Validation Accuracy: {round(valid_acc, r)}')



    E1.setup(X_train, t_train)
    E1.fit(t=t_train)
    print(f'--')
    MoE_train = E1.expert_score(X_train, t_train)
    MoE_valid = E1.expert_score(X_valid, t_valid)
    print(f'------')
    print()
    print(f'Gating: ')
    print(f'Training Accuracy: {round(MoE_train, r)}')
    print(f'Validation Accuracy: {round(MoE_valid, r)}')

    accuracies = []
    classes = np.unique(t_train)
    # sizes = [3, 5, 7]
    # num_layers = [1, 2]
    # sizes = [5, 7, 10]
    # num_layers = [1, 2]
    # sizes = [1, 7, 15]
    # num_layers = [1]

    sizes = [3, 7, 30]
    num_layers = [2]

    for s in sizes:
        for n in num_layers:
            sizes = [s] * n
            mlp_expert = MLPClassifier(hidden_layer_sizes=sizes, activation='relu')
            E1 = Expert(models, mlp_expert)
            E1.setup(X_train, t_train)
            i = 10
            for j in range(40):
                E1.partial_fit(t=t_train, n_iter=i)
                a = E1.expert_score(X_train, t_train)
                v = E1.expert_score(X_valid, t_valid)
                tot = i * (j + 1)
                # accuracies.append([tot, a, "training", f"{n}x{s}"])
                # accuracies.append([tot, v, "validation", f"{n}x{s}"])
                accuracies.append([tot, a, "training", s])
                accuracies.append([tot, v, "validation", s])
                # accuracies.append([tot, a, "training", n])
                # accuracies.append([tot, v, "validation", n])

    # ideal is 2x7
    # n-iter is 100

    acDF = pd.DataFrame(data=accuracies, columns=["n-iter", "score", "type", "size"])
    # acDF = pd.DataFrame(data=accuracies, columns=["n-iter", "score", "type", "depth"])

    sns.set_style("darkgrid")
    palette = sns.cubehelix_palette(light=.8, n_colors=4)
    # flare is red & orange, crest is green & blue
    sns.relplot(data=acDF, kind="line", x="n-iter", y="score", hue="size", style="type", linewidth=2, palette="flare")
    # sns.relplot(data=acDF, kind="line", x="n-iter", y="score", hue="depth", style="type", linewidth=2, palette="crest")





    plt.show()



