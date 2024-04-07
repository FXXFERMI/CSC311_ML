from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import data_cleaning as dc
import numpy as np
import seaborn as sns

seed = 42
np.random.seed(seed)

if __name__ == '__main__':

    t_train = dc.t_train
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


    # accuracies = []
    # for i in range(1, 10):
    #     dtree = DecisionTreeClassifier(max_depth=i)
    #     dtree.fit(X_train, t_train)
    #     a = dtree.score(X_train, t_train)
    #     v = dtree.score(X_valid, t_valid)
    #     accuracies.append([i, a, v])
    #
    # f, ax = plt.subplots(figsize=(6, 6))
    #
    # acDF = pd.DataFrame(data=accuracies, columns=["depth", "training", "validation"])
    # sns.set_color_codes("pastel")
    # sns.barplot(data=acDF, x="depth", y="training", label="Training", color="b")
    # sns.set_color_codes("muted")
    # sns.barplot(data=acDF, x="depth", y="validation", label="Validation", color="b")
    # for bar in ax.containers[0]:
    #     bar.set_alpha(1)
    # for bar in ax.containers[1]:
    #     bar.set_alpha(1)
    #     x = bar.get_x()
    #     center = x + 0.6 / 2
    #     bar.set_x(center - 0.2)
    #     bar.set_width(0.6)
    # ax.legend(ncol=2, loc="lower right", frameon=True)
    # ax.set(ylabel="Accuracy")
    #
    # # ---------------------------------------------
    #
    # accuracies = []
    # classes = np.unique(t_train)
    # # sizes = [3, 5, 7]
    # # num_layers = [1, 2]
    # # sizes = [5, 7, 10]
    # # num_layers = [1, 2]
    # # sizes = [1, 7, 15]
    # # num_layers = [1]
    #
    # sizes = [7]
    # num_layers = [2]
    #
    # for s in sizes:
    #     for n in num_layers:
    #         sizes = [s] * n
    #         mlp = MLPClassifier(hidden_layer_sizes=sizes, activation='relu')
    #         for i in range(200):
    #             mlp.partial_fit(X_train, t_train, classes=classes)
    #             if i % 10 == 0:
    #                 a = mlp.score(X_train, t_train)
    #                 v = mlp.score(X_valid, t_valid)
    #                 # accuracies.append([i, a, "training", f"{n}x{s}"])
    #                 # accuracies.append([i, v, "validation", f"{n}x{s}"])
    #                 # accuracies.append([i, a, "training", s])
    #                 # accuracies.append([i, v, "validation", s])
    #                 accuracies.append([i, a, "training", n])
    #                 accuracies.append([i, v, "validation", n])
    #
    # # ideal is 2x7
    # # n-iter is 100
    #
    # # acDF = pd.DataFrame(data=accuracies, columns=["n-iter", "score", "type", "size"])
    # acDF = pd.DataFrame(data=accuracies, columns=["n-iter", "score", "type", "depth"])
    #
    # sns.set_style("darkgrid")
    # palette = sns.cubehelix_palette(light=.8, n_colors=4)
    # # flare is red & orange, crest is green & blue
    # # sns.relplot(data=acDF, kind="line", x="n-iter", y="score", hue="size", style="type", linewidth=2, palette="flare")
    # sns.relplot(data=acDF, kind="line", x="n-iter", y="score", hue="depth", style="type", linewidth=2, palette="crest")

    accuracies = []
    strenghts = [1, 10, 100]
    # also tried 10, 50, low accuracy so not worth showing
    for i in range(10, 200, 10):
        print(i)
        for s in strenghts:
            lr = LogisticRegression(max_iter=i, C=s, solver='lbfgs')
            # lr.fit(X_train, t_train)
            multi_target_logreg = MultiOutputClassifier(lr, n_jobs=-1)
            multi_target_logreg.fit(X_train, t_train)

            ###### Prediction ####
            train_pre = multi_target_logreg.predict(X_train)
            correct_train_predictions = np.all(train_pre == t_train, axis=1)
            train_acc = np.mean(correct_train_predictions)

            a = multi_target_logreg.score(X_train, t_train)
            v = multi_target_logreg.score(X_valid, t_valid)
            # print(lr.n_iter_)
            accuracies.append([i, a, "training", s])
            accuracies.append([i, v, "validation", s])

    # n-iter = 100
    # s = 2

    acDF = pd.DataFrame(data=accuracies, columns=["n-iter", "score", "type", "strength"])
    # flare for close
    # Set2 for far
    sns.relplot(data=acDF, kind="line", x="n-iter", y="score", hue="strength", style="type", palette="Set2",
                linewidth=2)
    # sns.relplot(data=acDF, kind="line", x="n-iter", y="score", hue="strength", style="type", palette="Set2", linewidth=2)

    # # -- Best Validation Models! --
    #
    # dtree = DecisionTreeClassifier(max_depth=3)
    # dtree.fit(X_train, t_train)
    #
    # mlp = MLPClassifier(max_iter=100, hidden_layer_sizes=(7, 7), activation='relu')
    # mlp.fit(X_train, t_train)
    #
    # lr = LogisticRegression(max_iter=100, C=2)
    # lr.fit(X_train, t_train)
    #
    # print(f'-- Decision Tree --')
    # print(f'accuracy: {round(dtree.score(X_train, t_train), 4)}')
    # print(f'validation: {round(dtree.score(X_valid, t_valid), 4)}')
    #
    # print()
    # print(f'-- MLP --')
    # print(f'accuracy: {round(mlp.score(X_train, t_train), 4)}')
    # print(f'validation: {round(mlp.score(X_valid, t_valid), 4)}')

    print()
    print(f'-- Linear Regression --')
    print(f'accuracy: {round(multi_target_logreg.score(X_train, t_train), 4)}')
    print(f'validation: {round(multi_target_logreg.score(X_valid, t_valid), 4)}')

    plt.show()