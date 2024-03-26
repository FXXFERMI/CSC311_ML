import pandas
import seaborn as sns
import csv

import numpy as np
import re
import pandas as pd
import random as Random
#
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
#
import matplotlib as mpl
import matplotlib.pyplot as plt
#
# # https://seaborn.pydata.org/
import seaborn.objects as so







# def seaborn_1():
#     # Use a breakpoint in the code line below to debug your script.
#     df = sns.load_dataset("penguins")
#     sns.pairplot(df, hue="species")

def open_csv(directory):
    return open(directory, "r+")

file_name = "data/raw_data/clean_dataset.csv"
random_state = 42
def to_numeric(s):
    """Converts string `s` to a float.
    Invalid strings and NaN values will be converted to float('nan').
    """
    if isinstance(s, str):
        s = s.replace(",", '')
        s = pd.to_numeric(s, errors="coerce")
        return float(s)

def Q7_normalize_temp(s):
    if isinstance(s, str):
        s = s.replace(",", '')
        s = pd.to_numeric(s, errors="coerce")
        if float(s) >= 40:
            return 40
        elif float(s) <= -15:
            return -15
        else:
            return float(s)

def Q8_normalize_languages(s):
    if float(s) >= 15:
        return 15
    elif float(s) <= 0:
        return 0
    else:
        return float(s)

def Q9_normalize_fashion(s):
    if isinstance(s, str):
        s = s.replace(",", '')
        s = pd.to_numeric(s, errors="coerce")
        if float(s) >= 40:
            return 40
        elif float(s) <= 0:
            return 0
        else:
            return float(s)

def get_number_list(s):
    """Get a list of integers contained in string `s`
    """
    return [int(n) for n in re.findall("(\d+)", str(s))]

def get_number_list_clean(s):
    """Return a clean list of numbers contained in `s`.
    Additional cleaning includes removing numbers that are not of interest
    and standardizing return list size.
    """
    n_list = get_number_list(s)
    n_list += [-1]*(6-len(n_list))
    return n_list

def find_area_at_rank(l, i):
    """Return the area at a certain rank in list `l`.
    Areas are indexed starting at 1 as ordered in the survey.
    If area is not present in `l`, return -1.
    """
    return l.index(i) + 1 if i in l else -1

def cat_in_s(s, cat):
    """Return if a category is present in string `s` as an binary integer.
    """
    return int(cat in s) if not pd.isna(s) else 0

def handle_q6(l):
    "return 6 one-hot vectors for 6 ranks, 's'"

    return_square = []
    for i in range(len(l)):
        basic = [0] * 6
        basic[l[i] - 1] = 1
        return_square.append(basic)
    # print(return_square)
    # basic = [0] * 6
    # basic2 = [basic] * 6
    # # for i in range(len(l)):
    # #     basic[i][l[i] - 1] = 1
    # basic2[1][1] = 1

    return return_square

def eye_onehot(s):
    # print(s)
    targets = np.array(s)
    # print(targets)
    df2 = np.eye(8)[targets - 1]
    df3 = np.delete(df2, -1, axis=1)
    # print(df3)
    # print('-----')
    return df3

def get_column(c, s):
    return s[c]




if __name__ == "__main__":
    # Open csv
    df = pd.read_csv(file_name)

    # Normalize Q7-Q9
    df["Q7"] = df["Q7"].apply(Q7_normalize_temp).fillna(-1)
    df["Q8"] = df["Q8"].apply(Q8_normalize_languages).fillna(-1)
    df["Q9"] = df["Q9"].apply(Q9_normalize_fashion).fillna(-1)

    # Format Q6
    df["Q6_original"] = df["Q6"].apply(get_number_list_clean)

    # Move data from Q6_original into 6 different columns
    df["Q6_1"] = df["Q6_original"].str[0]
    df["Q6_2"] = df["Q6_original"].str[1]
    df["Q6_3"] = df["Q6_original"].str[2]
    df["Q6_4"] = df["Q6_original"].str[3]
    df["Q6_5"] = df["Q6_original"].str[4]
    df["Q6_6"] = df["Q6_original"].str[5]

    # shuffle data
    # *note that this is already done by train_test_split*
    seed = 123
    df = df.sample(frac=1, random_state=seed)

    # create datastack which will be used in training
    data_fets = np.stack([
        df["Q6_1"],
        df["Q6_2"],
        df["Q6_3"],
        df["Q6_4"],
        df["Q6_5"],
        df["Q6_6"],
        df["Q7"],
        df["Q8"],
        df["Q9"]
    ], axis=1)

    X = data_fets
    t = np.array(df["Label"])

    # Split into 8:1:1 train:valid:test (approx 1175 train, 145 valid, 145 test)
    validN = int(0.10 * X.shape[0])
    testN = int(0.10 * X.shape[0])

    X_tv, X_test, t_tv, t_test = train_test_split(X, t, test_size=testN, random_state=1)
    X_train, X_valid, t_train, t_valid = train_test_split(X_tv, t_tv, test_size=validN, random_state=1)

    # First, fit a decision tree
    tree = DecisionTreeClassifier(criterion="entropy", max_depth=6)
    tree.fit(X_train, t_train)

    # tuning hyper-parameters: depth
    # depths = [3, 4, 5, 6, 7, 8, 10, 50]
    # accuracies = []
    # for depth in range(1, 25, 1):
    #     tree = DecisionTreeClassifier(criterion="entropy", max_depth=depth)
    #     tree.fit(X_train, t_train)
    #     accuracies.append([depth, tree.score(X_train, t_train), "training"])
    #     accuracies.append([depth, tree.score(X_valid, t_valid), "validation"])
    #
    # acDF = pandas.DataFrame(data=accuracies, columns=["depth", "score", "type"])
    # sns.set_style("darkgrid")
    # sns.relplot(data = acDF, kind="line", x="depth", y="score", hue="type", dashes=False, marker="o")

    # best validation comes with depth 6
    print(f'Decision Tree: ')
    print("Training Accuracy:", tree.score(X_train, t_train))
    print("Validation Accuracy:", tree.score(X_valid, t_valid))
    print()



    # now we train an MLP
    mlp = MLPClassifier(hidden_layer_sizes=(10, 10), activation='relu', solver='sgd')
    # mlp.fit(X_train, t_train)

    # tune hyperparameters - num iterations, hidden layer sizes, num hidden layers
    accuracies = []
    classes = np.unique(t_train)
    for i in range(600):
        mlp.partial_fit(X_train, t_train, classes=classes)
        if i % 10 == 0:
            a = mlp.score(X_train, t_train)
            v = mlp.score(X_valid, t_valid)
            accuracies.append([i, a, "training"])
            accuracies.append([i, v, "validation"])

    acDF = pandas.DataFrame(data=accuracies, columns=["n-iter", "score", "type"])
    sns.relplot(data=acDF, kind="line", x="n-iter", y="score", style="type")

    # accuracies = []
    # classes = np.unique(t_train)
    # # sizes = [3, 5, 7, 10, 15, 20]
    # # num_layers = [1, 2, 3, 4, 5]
    # sizes = [15]
    # num_layers = [2]
    # for s in sizes:
    #     for n in num_layers:
    #         sizes = [s] * n
    #         mlp = MLPClassifier(hidden_layer_sizes=sizes, activation='relu', solver='sgd')
    #         for i in range(300):
    #             mlp.partial_fit(X_train, t_train, classes=classes)
    #             if i % 10 == 0:
    #                 a = mlp.score(X_train, t_train)
    #                 v = mlp.score(X_valid, t_valid)
    #                 accuracies.append([i, a, "training", f"{n}x{s}"])
    #                 accuracies.append([i, v, "validation", f"{n}x{s}"])
    # acDF = pandas.DataFrame(data=accuracies, columns=["n-iter", "score", "type", "size"])
    # sns.relplot(data = acDF, kind="line", x="n-iter", y="score", hue="size", style="type")
    #
    # best is 2 hidden, 10 wide, 150 iterations


    # show weight matricies as well
    all_coefs = mlp.coefs_[0]
    max = np.max(all_coefs, axis=1)
    min = np.min(all_coefs, axis=1)
    print(f'MLP: ')
    print(f'max is: {max.round(2)}')
    print(f'min is: {min.round(2)}')


    print("Training Accuracy:", mlp.score(X_train, t_train))
    print("Validation Accuracy:", mlp.score(X_valid, t_valid))
    print(f'')
    # accuracy becomes around 83%

    # plot Q7 across Q6
    # sns.catplot(data=df, x="Q6_6", hue="Label", y="Q7")

    # plot all values of Q6 in different axises
    # for i in range(1,7):
    #     x = f"Q6_{i}"
    #     sns.catplot(data=df, x=x, hue="Label", kind="count")
    #     plt.xlabel(column_names[i-1][3:])

    sns.catplot(data=df, x="Label", hue="Q6_1", kind="count")
    plt.xlabel("Skyscraper")
    sns.catplot(data=df, x="Label", hue="Q6_2", kind="count")
    plt.xlabel("Sport")
    sns.catplot(data=df, x="Label", hue="Q6_3", kind="count")
    plt.xlabel("Art & Music")
    sns.catplot(data=df, x="Label", hue="Q6_4", kind="count")
    plt.xlabel("Carnival")
    sns.catplot(data=df, x="Label", hue="Q6_5", kind="count")
    plt.xlabel("Cuisine")
    sns.catplot(data=df, x="Label", hue="Q6_6", kind="count")
    plt.xlabel("Economic")

    # show two-way graph
    sns.jointplot(data=df, x="Q7", y="Q9", hue="Label")

    # show distribution of Q7, Q8, and Q9
    sns.catplot(x=df["Label"], y=df["Q7"])
    sns.catplot(x=df["Label"], y=df["Q8"])
    sns.catplot(x=df["Label"], y=df["Q9"])
    plt.show()