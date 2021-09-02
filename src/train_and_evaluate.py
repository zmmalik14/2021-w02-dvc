import os
import math
import pandas as pd
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.metrics import average_precision_score
from numpyencoder import NumpyEncoder
from get_data import read_params
import argparse
import joblib
import json


def train_and_evaluate(config_path):
    """
    This function trains and evaluates a machine learning model on the dataset.

    :param config_path: the path of the config file to use
    """
    # Read configuration options
    config = read_params(config_path)
    model_type = config["train_and_evaluate"]["model"]
    test_data_path = config["split_data"]["test_path"]
    train_data_path = config["split_data"]["train_path"]
    random_state = config["base"]["random_state"]
    model_dir = config["model_dir"]
    target = [config["base"]["target_col"]]
    scores_file = config["reports"]["scores"]
    prc_file = config["reports"]["prc"]
    roc_file = config["reports"]["roc"]

    # Load training and validation datasets
    train = pd.read_csv(train_data_path, sep=",")
    test = pd.read_csv(test_data_path, sep=",")

    # Separate features (x) from label (y)
    train_y = train[target]
    test_y = test[target]
    train_x = train.drop(target, axis=1)
    test_x = test.drop(target, axis=1)

    if model_type == "logistic_regression":
        # Build logistic regression model
        model = LogisticRegression(solver='sag', random_state=random_state).fit(train_x, train_y)
    elif model_type == "random_forest":
        # Build random forest model
        model = RandomForestClassifier(n_estimators=50)
    else:
        return

    # Fit the model to the training data
    model.fit(train_x, train_y)

    # Report training set score
    train_score = model.score(train_x, train_y) * 100
    print(train_score)
    # Report test set score
    test_score = model.score(test_x, test_y) * 100
    print(test_score)

    # Predict output for observations in validation set
    predicted_val = model.predict(test_x)

    # Calculate performance metrics
    precision, recall, prc_thresholds = metrics.precision_recall_curve(test_y, predicted_val)
    fpr, tpr, roc_thresholds = metrics.roc_curve(test_y, predicted_val)
    nth_point = math.ceil(len(prc_thresholds) / 1000)
    prc_points = list(zip(precision, recall, prc_thresholds))[::nth_point]

    with open(prc_file, "w") as fd:
        prcs = {
            "prc": [
                {"precision": p, "recall": r, "threshold": t}
                for p, r, t in prc_points
            ]
        }
        json.dump(prcs, fd, indent=4, cls=NumpyEncoder)

    with open(roc_file, "w") as fd:
        rocs = {
            "roc": [
                {"fpr": fp, "tpr": tp, "threshold": t}
                for fp, tp, t in zip(fpr, tpr, roc_thresholds)
            ]
        }
        json.dump(rocs, fd, indent=4, cls=NumpyEncoder)

    # Print classification report
    print(classification_report(test_y, predicted_val))

    # Confusion Matrix and plot
    cm = confusion_matrix(test_y, predicted_val)
    print(cm)
    df1 = pd.DataFrame(predicted_val, columns=['Predicted'])
    df_cm = pd.concat([test_y, df1], axis=1)
    print(df_cm)
    df_cm.to_csv('cm.csv', index=False)

    # Receiver operating characteristic - area under curve
    roc_auc = roc_auc_score(test_y, model.predict_proba(test_x)[:, 1])
    print('ROC_AUC:{0:0.2f}'.format(roc_auc))

    # Model accuracy
    model_accuracy = accuracy_score(test_y, predicted_val)
    print('Model Accuracy:{0:0.2f}'.format(model_accuracy))

    # Average precision score
    average_precision = average_precision_score(test_y, predicted_val)
    print('Average precision-recall score: {0:0.2f}'.format(average_precision))

    with open(scores_file, "w") as f:
        scores = {
            "train_score": train_score,
            "test_score": test_score,
            "roc_auc": roc_auc,
            "Precision": list(precision),
            "Recall": list(recall),
            "Average precision": average_precision,
            "Model Accuracy": model_accuracy
        }
        json.dump(scores, f, indent=4)

    # Output model
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")
    joblib.dump(model, model_path)


if __name__ == "__main__":
    # If the file is being run, parse command line arguments to get config filepath
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()

    # Train and evaluate the model
    train_and_evaluate(config_path=parsed_args.config)
