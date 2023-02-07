import os
import time
import numpy as np
import wandb

from sdg_classifier.metrics.metrics import (
    exact_match_ratio, hamming_score, hamming_loss,
    precision_overall, recall_overall, f1_overall
)


# define folder for artifact persistence
def get_run_logdir(root_logdir):
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)


# To run this function, you must log in to your WandB account
def log_to_wandb(model, valid_set, test_set, bce, accuracy, valid_bce,
                 valid_accuracy, model_dir, save_model=True):

    if save_model:
        model.save(model_dir)

    # Logging metrics
    y_pred = ((model.predict(test_set) > 0.5) + 0)
    for i, (X, y) in enumerate(test_set):
        if i > 0:
            y_true = np.concatenate((y_true, y.numpy()))
        else:
            y_true = y.numpy()

    em_ratio = exact_match_ratio(y_true, y_pred)
    overall_accuracy = hamming_score(y_true, y_pred)
    overall_loss = hamming_loss(y_true, y_pred)
    precision = precision_overall(y_true, y_pred)
    recall = recall_overall(y_true, y_pred)
    f1 = f1_overall(y_true, y_pred)

    wandb.log({
        "Accuracy": accuracy,
        "Validation Accuracy": valid_accuracy,
        "Loss": bce,
        "Validation Loss": valid_bce,
        "Exact Match Ratio": em_ratio,
        "Hamming Score": overall_accuracy,
        "Hamming Loss": overall_loss,
        "Overall Precision": precision,
        "Overall Recall": recall,
        "Overall F1": f1
    })
