import numpy as np
import random
from pandas import DataFrame
from asgard.metrics.metrics import *
import pytest

RANDOM_SEED = 42

np.random.seed(seed=RANDOM_SEED)
random.seed(42)

y_pred = np.random.randint(2, size=(10, 4))
y_true = np.random.randint(2, size=(10, 4))
labels = [f"SDG{i}" for i in range(1, 5)]
print_multilabel_metrics(y_true, y_pred)

def test_compute_binary_metrics():
    binary_metrics = compute_binary_metrics(y_true, y_pred, labels=labels)

    acc = binary_metrics.loc["Accuracy", :].values
    rec = binary_metrics.loc["Recall", :].values
    prec = binary_metrics.loc["Precision", :].values
    f1 = binary_metrics.loc["F1 Score", :].values
    roc = binary_metrics.loc["ROC AUC", :].values

    assert isinstance(binary_metrics, DataFrame)
    assert (acc == np.array([0.5, 0.6, 0.3, 0.2])).all()
    assert (rec == np.array([0.6, 0.7143, 0.25, 0.])).all()
    assert (prec == np.array([0.5, 0.7143, 0.20, 0.])).all()
    assert (f1 == np.array([0.5455, 0.7143, 0.2222, 0.])).all()
    assert (roc == np.array([0.5, 0.5238, 0.2917, 0.25])).all()


def test_exact_match_ratio():
    y_true_em = y_true.copy()
    y_pred_em = y_pred.copy()

    y_true_em[0, :] = [1, 1, 0, 1]
    y_pred_em[0, :] = [1, 1, 0, 1]

    em_ratio = exact_match_ratio(y_true_em, y_pred_em)
    assert em_ratio == 0.1


def test_hamming_score():
    hamm_score = hamming_score(y_true, y_pred)

    assert np.round(hamm_score, 5) == 0.25833


def test_hamming_loss():
    hamm_loss = hamming_loss(y_true, y_pred)

    assert hamm_loss == 0.6


def test_precision_overall():
    prec_overall = precision_overall(y_true, y_pred)

    assert np.round(prec_overall, 5) == 0.51667


def test_recall_overall():
    rec_overall = recall_overall(y_true, y_pred)

    assert rec_overall == 0.4


def test_f1_overall():
    f1 = f1_overall(y_true, y_pred)

    assert np.round(f1, 5) == 0.37333
