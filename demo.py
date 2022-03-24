import os
import joblib
import pandas as pd

DATASET_FILE_PATH = "demo-dataset.csv"
# DATASET_FILE_PATH = "final-merged-dataset.csv"


def preprocess(df):
    # 1. Remove the strings columnns from dataset.
    df.drop(
        [
            "Address A",
            "Address B",
        ],
        axis=1,
        inplace=True,
    )
    print("Dropped strings...", df.size)

    # 2. Remove 'Unnamed' column from dataset. This comes from reading CSV.
    # https://stackoverflow.com/questions/43983622/remove-unnamed-columns-in-pandas-dataframe
    df.drop(
        df.columns[df.columns.str.contains("unnamed", case=False)], axis=1, inplace=True
    )
    print("Dropped Unamed column...", df.size)

    # 3. Remove the invalid rows ie axis=0 from the dataset.
    df.dropna(axis=0, inplace=True)
    print("Dropped NA values...", df.size)

    # 4. After deleting the rows, shuffle and delete the index column.
    df = df.sample(frac=1).reset_index(drop=True)
    print("Shuffling Resetting index...", df.size)

    # 5. Get the result(quic) column values and remove dataset.
    targets = list(df["Target"].values)
    df.drop(df.columns[len(df.columns) - 1], axis=1, inplace=True)
    print("Dropped targets column...", df.size)

    return df, targets


def is_quic(packet):
    if packet:
        return "quic"
    else:
        return "non-quic"


if __name__ == "__main__":

    dataset = pd.read_csv(DATASET_FILE_PATH)
    print("Dataset preprocess ...\n")
    X_test, Y_test = preprocess(dataset)
    print("\nDataset preprocess completed...\n")

    logistic_regression_path = os.path.join("models", "LogisticRegression")
    gradient_boosting_classifier_path = os.path.join("models", "GradientBoost")
    decision_tree_classifier_path = os.path.join("models", "DecisionTree")
    k_neighbors_classifier_path = os.path.join("models", "KNN")
    random_forest_classifier_path = os.path.join("models", "RandomForest")

    print("Loading trained models...\n")
    logistic_regression_model = joblib.load(logistic_regression_path)
    gradient_boosting_classifier_model = joblib.load(gradient_boosting_classifier_path)
    decision_tree_classifier_model = joblib.load(decision_tree_classifier_path)
    k_neighbors_classifier_model = joblib.load(k_neighbors_classifier_path)
    random_forest_classifier_model = joblib.load(random_forest_classifier_path)

    print("Accuracy...\n")
    print(
        "{0:<35} {1}%".format(
            "Logistic_Regression_Model",
            round(logistic_regression_model.score(X_test, Y_test) * 100, 2),
        )
    )
    print(
        "{0:<35} {1}%".format(
            "Gradient_Boosting_Classifier_Model",
            round(gradient_boosting_classifier_model.score(X_test, Y_test) * 100, 2),
        )
    )
    print(
        "{0:<35} {1}%".format(
            "Decision_Tree_Classifier_Model",
            round(decision_tree_classifier_model.score(X_test, Y_test) * 100, 2),
        )
    )

    print(
        "{0:<35} {1}%".format(
            "K_Neighbors_Classifier_Model",
            round(k_neighbors_classifier_model.score(X_test, Y_test) * 100, 2),
        )
    )

    print(
        "{0:<35} {1}%".format(
            "Random_Forest_Classifier_Model",
            round(random_forest_classifier_model.score(X_test, Y_test) * 100, 2),
        )
    )

    print("\nPrediction...\n")
    logistic_regression_prediction = logistic_regression_model.predict(X_test)
    gradient_boosting_classifier_prediction = gradient_boosting_classifier_model.predict(
        X_test
    )
    decision_tree_classifier_prediction = decision_tree_classifier_model.predict(X_test)
    k_neighbors_classifier_prediction = k_neighbors_classifier_model.predict(X_test)
    random_forest_classifier_prediction = random_forest_classifier_model.predict(X_test)

    print(
        "{0:<20} {1:<28} {2:<28} {3:<28} {4:<28} {5:<28} {6:<28}".format(
            "Packet Time",
            "Logistic_Regression",
            "Gradient_Boosting_Classifier",
            "Decision_Tree_Classifier",
            "K_Neighbors_Classifier",
            "Random_Forest_Classifier",
            "Actual Packet Protocol",
        )
    )

    predict_miss_count = {
        "Logistic_Regression_Model": 0,
        "Gradient_Boosting_Classifier_Model": 0,
        "Decision_Tree_Classifier_Model": 0,
        "K_Neighbors_Classifier_Model": 0,
        "Random_Forest_Classifier_Model": 0,
    }
    for index, test in enumerate(X_test.values):
        if logistic_regression_prediction[index] != Y_test[index]:
            predict_miss_count["Logistic_Regression_Model"] += 1
        if gradient_boosting_classifier_prediction[index] != Y_test[index]:
            predict_miss_count["Gradient_Boosting_Classifier_Model"] += 1
        if decision_tree_classifier_prediction[index] != Y_test[index]:
            predict_miss_count["Decision_Tree_Classifier_Model"] += 1
        if k_neighbors_classifier_prediction[index] != Y_test[index]:
            predict_miss_count["K_Neighbors_Classifier_Model"] += 1
        if random_forest_classifier_prediction[index] != Y_test[index]:
            predict_miss_count["Random_Forest_Classifier_Model"] += 1

        print(
            "{0:<20} {1:<28} {2:<28} {3:<28} {4:<28} {5:<28} {6:<28}".format(
                "{0}-{1}".format(index, test[1]),
                is_quic(logistic_regression_prediction[index]),
                is_quic(gradient_boosting_classifier_prediction[index]),
                is_quic(decision_tree_classifier_prediction[index]),
                is_quic(k_neighbors_classifier_prediction[index]),
                is_quic(random_forest_classifier_prediction[index]),
                is_quic(Y_test[index]),
            )
        )

    print("\nPrediction miss count...\n")
    for k, v in predict_miss_count.items():
        print("{0:<38} {1}/{2}".format(k, v, len(X_test)))
