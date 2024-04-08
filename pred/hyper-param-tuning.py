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


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)  # only difference

def pick_random(x):
    a = x.sample(n=1, weights=x)
    return a.index.to_list()[0]


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


    def generate_t(self, X_train, t_train):
        """
        Produces a vector indicating which expert we should listen to!
        self.predictions shows the predictions for the given X_train value.
        """
        self.classes = np.unique(t_train)

        self.predictions = self.small_predictions(X_train)
        targets = np.transpose(np.array([t_train == self.predictions[:, i] for i in range(len(models))]).astype('int64'))
        return targets





    def expert_score(self, X, t):
        """
        Transforms the original input into the predictions from experts.
        Then, scores the expert based on this new input.
        """
        weights = self.predict_weights(X)
        standard_error = np.sum(np.square((weights - t))) / (weights.shape[0] * weights.shape[1])
        return standard_error

    # def softmax(self, x):
    #     """Compute softmax values for each sets of scores in x."""
    #     e_x = np.exp(x - np.max(x))
    #     return e_x / e_x.sum(axis=0)  # only difference

    def score(self, X, t_actual):
        """
        Figure out the results after deferring to an expert (deferee)
        Calculate how often our model was correct after deferring to that expert (correct)
        Figure out the accuracy (acc)
        """
        deferee = self.defer_to_expert(X)
        correct = np.count_nonzero(t_actual.astype('int64') == deferee.astype('int64'))
        acc = correct / t_actual.shape[0]

        return acc


    def defer_to_expert(self, X):
        """
        Our main model will pick which model to use for each input.
        We will then return the value that the given model is suggesting.

        Weights represents how badly the model wants to use that expert's prediction.
        Indices represents the index of the expert the model wants to defer to
        """
        self.predictions = self.small_predictions(X)
        # weights = self.predict_weights(X)
        weights = pd.DataFrame(self.predict_weights(X))
        weights = weights.apply(softmax, axis=1)
        # print(weights[:1])
        # print(weights[:1].sample(n=1, weights=weights[0], axis=1))
        selection = weights.apply(pick_random, axis=1)
        indices = selection.to_numpy()

        # indices = np.array(np.argmax(weights, axis=1))

        predictions = np.array([self.predictions[i][j] for i, j in enumerate(indices)])
        return predictions.astype('int64')

    def predict_weights(self, X):
        weights = self.expert.predict(X)
        return weights

    def format_X(self, X=None):
        pass




    def fit(self, X, t):
        self.expert.fit(X, t)

        # self.expert.fit(predictions, t)

    def partial_fit(self, X, t, n_iter=100):
        """
        Partially fits the model based on the number of iterations
        """

        for i in range(n_iter):
            self.expert.partial_fit(X, t)







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
    t_train2 = E1.generate_t(X_train[:10], t_train[:10])
    E1.partial_fit(X=X_train[:10], t=t_train2)
    E1.predict_weights(X_train[:10])

    print(f'score')
    print(E1.score(X_train[:10], t_train[:10]))
    print(t_train2)
    print(E1.defer_to_expert(X_train[:10]))
    print(t_train[:10])


    print()
    print(f'Updated MoE:')
    MoE_train = E1.expert_score(X_train[:10], t_train2)

    t_trainV = E1.generate_t(X_valid[:10], t_valid[:10])
    MoE_valid = E1.expert_score(X_valid[:10], t_trainV)
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
            mlp_expert = MLPRegressor(hidden_layer_sizes=sizes, activation='relu')
            E1 = Expert(models, mlp_expert)
            tE_train = E1.generate_t(X_train, t_train)
            tE_valid = E1.generate_t(X_valid, t_valid)

            i = 5
            for j in range(40):
                E1.partial_fit(X=X_train, t=tE_train, n_iter=i)
                # a = E1.expert_score(X_train, tE_train)
                # v = E1.expert_score(X_valid, tE_valid)
                a = E1.score(X_train, t_train)
                v = E1.score(X_valid, t_valid)
                tot = i * (j + 1)
                print(tot)
                # accuracies.append([tot, a, "training", f"{n}x{s}"])
                # accuracies.append([tot, v, "validation", f"{n}x{s}"])
                accuracies.append([tot, a, "training", s])
                accuracies.append([tot, v, "validation", s])
                # accuracies.append([tot, a, "training", n])
                # accuracies.append([tot, v, "validation", n])

    # ideal is 2x7
    # n-iter is 65

    # print(E1.score(X_test, t_test))



    acDF = pd.DataFrame(data=accuracies, columns=["n-iter", "score", "type", "size"])
    # acDF = pd.DataFrame(data=accuracies, columns=["n-iter", "score", "type", "depth"])

    sns.set_style("darkgrid")
    palette = sns.cubehelix_palette(light=.8, n_colors=4)
    # flare is red & orange, crest is green & blue
    sns.relplot(data=acDF, kind="line", x="n-iter", y="score", hue="size", style="type", linewidth=2, palette="flare")
    # sns.relplot(data=acDF, kind="line", x="n-iter", y="score", hue="depth", style="type", linewidth=2, palette="crest")





    plt.show()



