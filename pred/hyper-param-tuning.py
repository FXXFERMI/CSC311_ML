from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
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

    def small_predictions(self, X):
        """
        Returns the prediction for every datapoint for every model.
        size is (X.shape[0], # models)
        """
        # predictions = np.zeros((X.shape[0], len(self.models) + X.shape[1]))
        predictions = np.zeros((X.shape[0], len(self.models)))

        for i, model in enumerate(self.models):
            predictions[:, i] = model.predict(X)
        # predictions = np.concatenate((predictions, X), axis=1)
        # predictions[:, i+1] = X
        return predictions

    def setup(self, X_train, t_train):
        self.predictions = self.small_predictions(X_train)
        self.classes = np.unique(t_train)

        # print(self.predictions[:, 0])
        # print(self.predictions[:, 1])
        # print(self.predictions[:, 2])
        # print(t)
        self.targets = np.transpose(np.array([t_train == self.predictions[:, i] for i in range(len(models))]).astype('int64'))
        # print(self.targets)
        self.X = X_train
        self.t = t_train


    def expert_predict(self, X):
        """
        Turns the original very many feature input
        into the prediction of each expert.
        Then, performs the prediction!
        """
        predictions = self.small_predictions(X)
        return self.expert.predict_weights(predictions)

    def expert_score(self):
        """
        Transforms the original input into the predictions from experts.
        Then, scores the expert based on this new input.
        """
        # score = self.expert.score(self.X, self.targets)
        prediction = self.predict_weights(self.X)
        standard_error = np.sum(np.square((prediction - self.targets))) / (prediction.shape[0] * prediction.shape[1])
        return standard_error

    # def softmax(self, x):
    #     """Compute softmax values for each sets of scores in x."""
    #     e_x = np.exp(x - np.max(x))
    #     return e_x / e_x.sum(axis=0)  # only difference

    def score(self, X, t):
        predictions = self.expert_predict()
        correct = np.count_nonzero(self.t.astype('int64') == predictions.astype('int64'))
        acc = correct / self.t.shape[0]

        return acc

    def expert_predict(self):
        weights = self.predict_weights(self.X)
        indices = np.array(np.argmax(weights, axis=1))
        predictions = np.array([self.predictions[i][j] for i, j in enumerate(indices)])
        return predictions.astype('int64')

    def predict_weights(self, X):
        weights = self.expert.predict(X)
        return weights

    def format_X(self, X=None):
        if X == None:
            X = self.X
        else:
            X = X




    def fit(self):
        self.expert.fit(self.X, self.targets)

        # self.expert.fit(predictions, t)

    def partial_fit(self, n_iter=100):
        """
        Partially fits the model based on the number of iterations
        """

        for i in range(n_iter):
            self.expert.partial_fit(self.X, self.targets)







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
    expert = MLPRegressor(max_iter=100, hidden_layer_sizes=(7, 7), activation='relu')
    # expert = LogisticRegression(max_iter=100, C=2)
    E1 = Expert(models, expert)
    E1.setup(X_train[:10], t_train[:10])
    E1.partial_fit(10)
    E1.predict_weights(X_train[:10])

    print(f'score')
    print(E1.score(X_train[:10], t_train[:10]))


    print()
    print(f'Updated MoE:')
    MoE_train = E1.expert_score()
    MoE_valid = E1.expert_score()
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

    sizes = [7]
    num_layers = [2]

    for s in sizes:
        for n in num_layers:
            sizes = [s] * n
            mlp_expert = MLPRegressor(max_iter=100, hidden_layer_sizes=(7, 7), activation='relu')
            E1 = Expert(models, mlp_expert)
            E1.setup(X_train, t_train)


            i = 10
            for j in range(40):
                E1.partial_fit(n_iter=i)
                a = E1.expert_score()
                v = E1.expert_score()
                tot = i * (j + 1)
                print(tot)
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



