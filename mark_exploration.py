import pandas
import seaborn as sns
import csv

import numpy as np
import re
import pandas as pd
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

file_name = "clean_dataset.csv"
random_state = 42
def to_numeric(s):
    """Converts string `s` to a float.
    Invalid strings and NaN values will be converted to float('nan').
    """
    if isinstance(s, str):
        s = s.replace(",", '')
        s = pd.to_numeric(s, errors="coerce")
        if float(s) >= 50:
            return pd.isna(s)
        else:
            return float(s)

def adjust_languages(s):
    if float(s) >= 50:
        return pd.isna(s)
    else:
        return float(s)

def adjust_fashion(s):
    if isinstance(s, str):
        s = s.replace(",", '')
        s = pd.to_numeric(s, errors="coerce")
        if float(s) >= 50:
            return pd.isna(s)
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

def get_number(s):
    """Get the first number contained in string `s`.
    If `s` does not contain any numbers, return -1.
    """
    n_list = get_number_list(s)
    return n_list[0] if len(n_list) >= 1 else -1

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
    print("hello world")

    # reader = open_csv("clean_dataset.csv")
    df = pd.read_csv("clean_dataset.csv")
    df["Q7"] = df["Q7"].apply(to_numeric).fillna(0)
    df["Q8"] = df["Q8"].apply(adjust_languages).fillna(0)
    df["Q9"] = df["Q9"].apply(adjust_fashion).fillna(0)
    # Clean for number categories
    df["Q1"] = df["Q1"].apply(get_number)
    # Create area rank categories

    df["Q6_original"] = df["Q6"].apply(get_number_list_clean)
    df["Q6"] = df["Q6"].apply(get_number_list_clean)
    # print(df["Q6"])
    # print(df["Q6"][1])

    # ----- my code starts
    # why not just have 6 one-hot vectors!
    #
    column_names = ["Q6_skyscraper", "Q6_sport", "Q6_art", "Q6_carnival", "Q6_cuisine", "Q6_economic"]

    # handle_q6([6, 5, 4, 3, 2, 1])

    # print(df["Q6"])

    # print('---')
    df["Q6"] = df["Q6"].apply(eye_onehot)
    # print("---")
    # print(df["Q6"])
    # print(df["Q6"][1])

    print("-----")
    for i in range(len(column_names)):
        print(f'column {i}')
        df[column_names[i]] = df["Q6"].apply(lambda s: get_column(i, s))
    # print(df["Q6_skyscraper"])


    # ------- my code ends

    print('----------')
    print(list(df))
    print(df["Q6_original"])

    df["Q6_1"] = df["Q6_original"].str[0]
    df["Q6_2"] = df["Q6_original"].str[1]
    df["Q6_3"] = df["Q6_original"].str[2]
    df["Q6_4"] = df["Q6_original"].str[3]
    df["Q6_5"] = df["Q6_original"].str[4]
    df["Q6_6"] = df["Q6_original"].str[5]
    # df["Q6_0"] = df["Q6_original"].str[0]

    titanic = sns.load_dataset("titanic")
    print(titanic)

    # differentiate btwn dubai & NYC!
    # Both very popular Q6 and Q1.
    # Both very different temps

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split

    data_fets = np.stack([
        # gender_female: this code creates an array of booleans, which converted into 0 and 1
        df["Q6_1"],
        # re_hispanic: this code leverages addition to perform an "or" operation
        df["Q6_2"],
        # re_white
        df["Q6_3"],
        # re_black
        df["Q6_4"],
        # re_aisan
        df["Q6_5"],
        # chest_pain_ever
        df["Q6_6"],
        # drink_alcohol
        df["Q7"],
        # age: this is a numeric value and no transformations are required
        df["Q8"],
        # blood_cholesterol
        df["Q9"]
    ], axis=1)

    print(data_fets.shape)
    X = data_fets
    t = np.array(df["Label"])

    # First, we will use `train_test_split` to split the data set into
    # 900 training+validation, and 500 test:
    X_tv, X_test, t_tv, t_test = train_test_split(X, t, test_size=500 / 1468, random_state=1)

    # Then, use `train_test_split` to split the training+validation data
    # into 5000 train and 1500 validation
    X_train, X_valid, t_train, t_valid = train_test_split(X_tv, t_tv, test_size=250 / 968, random_state=1)

    tree = DecisionTreeClassifier(criterion="entropy", max_depth=5)
    tree.fit(X_train, t_train)

    print("Training Accuracy:", tree.score(X_train, t_train))
    print("Validation Accuracy:", tree.score(X_valid, t_valid))
    # accuracy becomes around 85%

    mlp = MLPClassifier(hidden_layer_sizes=(10, 10), activation='relu', solver='sgd')
    # mlp.fit(X_train, t_train)

    accuracies = []
    classes = np.unique(t_train)
    for i in range(600):
        mlp.partial_fit(X_train, t_train, classes=classes)
        if i % 10 == 0:
            a = mlp.score(X_train, t_train)
            v = mlp.score(X_valid, t_valid)
            accuracies.append([i, a, "training"])
            accuracies.append([i, v, "validation"])

    accuracies = []
    classes = np.unique(t_train)
    # sizes = [3, 5, 7, 10, 15, 20]
    # num_layers = [1, 2, 3, 4, 5]
    sizes = [15]
    num_layers = [2]
    for s in sizes:
        for n in num_layers:
            sizes = [s] * n
            mlp = MLPClassifier(hidden_layer_sizes=sizes, activation='relu', solver='sgd')
            for i in range(300):
                mlp.partial_fit(X_train, t_train, classes=classes)
                if i % 10 == 0:
                    a = mlp.score(X_train, t_train)
                    v = mlp.score(X_valid, t_valid)
                    accuracies.append([i, a, "training", f"{n}x{s}"])
                    accuracies.append([i, v, "validation", f"{n}x{s}"])


    acDF = pandas.DataFrame(data=accuracies, columns=["n-iter", "score", "type", "size"])

    sns.set_style("darkgrid")
    palette = sns.cubehelix_palette(light=.8, n_colors=4)
    sns.relplot(data = acDF, kind="line", x="n-iter", y="score", hue="size", style="type")

    all_coefs = mlp.coefs_[0]

    max = np.max(all_coefs, axis=1)
    min = np.min(all_coefs, axis=1)
    print(f'max is: {max.round(2)}')
    print(f'min is: {min.round(2)}')


    print("Training Accuracy:", mlp.score(X_train, t_train))
    print("Validation Accuracy:", mlp.score(X_valid, t_valid))
    # accuracy becomes around 85%


    # depths = [5, 10, 15, 20, 50, 100]
    # # depths = [5, 10]
    # accuracies = []
    # for depth in range(1, 25, 1):
    #     tree = DecisionTreeClassifier(criterion="entropy", max_depth=depth)
    #     tree.fit(X_train, t_train)
    #     accuracies.append([depth, tree.score(X_train, t_train), "training"])
    #     accuracies.append([depth, tree.score(X_valid, t_valid), "validation"])
    #
    # acDF = pandas.DataFrame(data=accuracies, columns=["depth", "score", "type"])
    #
    # sns.set_style("darkgrid")
    # sns.relplot(data = acDF, kind="line", x="depth", y="score", hue="type", dashes=False, marker="o")

    # best depth is 5









    # sns.catplot(data=df, x="Q6_6", hue="Label", y="Q7")


    # for i in range(1,7):
    #     x = f"Q6_{i}"
    #     sns.catplot(data=df, x=x, hue="Label", kind="count")
    #     plt.xlabel(column_names[i-1][3:])

    # sns.catplot(data=df, x="Label", hue="Q6_1", kind="count")
    # plt.xlabel("Skyscraper")

    # sns.catplot(data=df, x="Label", hue="Q6_2", kind="count")
    # plt.xlabel("Sport")
    # sns.catplot(data=df, x="Label", hue="Q6_3", kind="count")
    # plt.xlabel("Art & Music")
    # sns.catplot(data=df, x="Label", hue="Q6_4", kind="count")
    # plt.xlabel("Carnival")
    # sns.catplot(data=df, x="Label", hue="Q6_5", kind="count")
    # plt.xlabel("Cuisine")
    # sns.catplot(data=df, x="Label", hue="Q6_6", kind="count")
    # plt.xlabel("Economic")

    # sns.catplot(data=df, x="Label", hue="Q6_0", kind="count")

    # sns.catplot(data=df, x="Label", hue="Q6_4", kind="count")
    # sns.jointplot(data=df, x="Q7", y="Q8", hue="Label")
    # sns.catplot(x=df["Label"], y=df["Q7"])
    plt.show()