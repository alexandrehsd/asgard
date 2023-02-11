from pprint import pprint

import numpy as np
import pandas as pd
from sklearn.metrics import (  # isort:skip
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_binary_metrics(y_test, y_preds, labels):
    metrics = {}.fromkeys(labels, None)
    for label in labels:
        metrics[label] = {}.fromkeys(
            ["Accuracy", "Recall", "Precision", "F1 Score", "ROC AUC"], None
        )

    for i, label in enumerate(labels):
        y_true = y_test[:, i]
        y_pred = y_preds[:, i].round()

        metrics[label]["Accuracy"] = accuracy_score(y_true, y_pred)
        metrics[label]["Recall"] = recall_score(y_true, y_pred)
        metrics[label]["Precision"] = precision_score(y_true, y_pred)
        metrics[label]["F1 Score"] = f1_score(y_true, y_pred)
        metrics[label]["ROC AUC"] = roc_auc_score(y_true, y_pred)

    metrics_data = pd.DataFrame(metrics).round(4)
    return metrics_data


def exact_match_ratio(y_true, y_pred):
    return np.all(y_pred == y_true, axis=1).mean()


# also called overall accuracy
def hamming_score(y_true, y_pred):
    accum = 0
    for i in range(y_true.shape[0]):
        accum += np.sum(np.logical_and(y_true[i], y_pred[i])) / np.sum(
            np.logical_or(y_true[i], y_pred[i])
        )
    return accum / y_true.shape[0]


# also called overall loss
def hamming_loss(y_true, y_pred):
    accum = 0
    for i in range(y_true.shape[0]):
        accum += np.size(y_true[i] == y_pred[i]) - np.count_nonzero(
            y_true[i] == y_pred[i]
        )
    return accum / (y_true.shape[0] * y_true.shape[1])


def precision_overall(y_true, y_pred):
    accum = 0
    for i in range(y_true.shape[0]):
        if np.sum(y_pred[i]) == 0:
            continue
        accum += np.sum(np.logical_and(y_true[i], y_pred[i])) / np.sum(y_pred[i])
    return accum / y_true.shape[0]


def recall_overall(y_true, y_pred):
    accum = 0
    for i in range(y_true.shape[0]):
        if np.sum(y_true[i]) == 0:
            continue
        accum += np.sum(np.logical_and(y_true[i], y_pred[i])) / np.sum(y_true[i])
    return accum / y_true.shape[0]


def f1_overall(y_true, y_pred):
    accum = 0
    for i in range(y_true.shape[0]):
        if (np.sum(y_true[i]) == 0) and (np.sum(y_pred[i]) == 0):
            continue
        accum += (2 * np.sum(np.logical_and(y_true[i], y_pred[i]))) / (
            np.sum(y_true[i]) + np.sum(y_pred[i])
        )
    return accum / y_true.shape[0]


def print_multilabel_metrics(y_true, y_pred):
    em_ratio = exact_match_ratio(y_true, y_pred)
    overall_accuracy = hamming_score(y_true, y_pred)
    overall_loss = hamming_loss(y_true, y_pred)
    precision = precision_overall(y_true, y_pred)
    recall = recall_overall(y_true, y_pred)
    f1 = f1_overall(y_true, y_pred)

    pprint(
        {
            "Exact Match": np.round(em_ratio, 4),
            "Overall Accuracy": np.round(overall_accuracy, 4),
            "Overall Loss": np.round(overall_loss, 4),
            "Overall Precision": np.round(precision, 4),
            "Overall Recall": np.round(recall, 4),
            "Overall F1-Score": np.round(f1, 4),
        }
    )
