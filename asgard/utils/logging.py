import time

import numpy as np
import wandb

from asgard.metrics.metrics import (  # isort:skip
    exact_match_ratio,
    f1_overall,
    hamming_loss,
    hamming_score,
    precision_overall,
    recall_overall,
)


# define folder for artifact persistence
def get_run_id():
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return run_id


# To run this function, you must log in to your WandB account
def log_to_wandb(model, valid_set, test_set, emissions_tracker, model_architecture):
    bce, accuracy = model.evaluate(test_set)
    valid_bce, valid_accuracy = model.evaluate(valid_set)

    # Logging metrics
    y_pred = (model.predict(test_set) > 0.5) + 0
    y_true = np.concatenate([y.numpy() for X, y in test_set])

    em_ratio = exact_match_ratio(y_true, y_pred)
    overall_accuracy = hamming_score(y_true, y_pred)
    overall_loss = hamming_loss(y_true, y_pred)
    precision = precision_overall(y_true, y_pred)
    recall = recall_overall(y_true, y_pred)
    f1 = f1_overall(y_true, y_pred)

    wandb.log(
        {
            "Accuracy": accuracy,
            "Validation Accuracy": valid_accuracy,
            "Loss": bce,
            "Validation Loss": valid_bce,
            "Exact Match Ratio": em_ratio,
            "Hamming Score": overall_accuracy,
            "Hamming Loss": overall_loss,
            "Overall Precision": precision,
            "Overall Recall": recall,
            "Overall F1": f1,
            "Number of Parameters": model.count_params(),
            "Energy Consumed": emissions_tracker.final_emissions_data.energy_consumed,
            "Energy RAM": emissions_tracker.final_emissions_data.ram_energy,
            "Energy GPU": emissions_tracker.final_emissions_data.gpu_energy,
            "Energy CPU": emissions_tracker.final_emissions_data.cpu_energy,
            "Co2 Emissions": emissions_tracker.final_emissions_data.emissions,
            "Co2 Emissions Rate": emissions_tracker.final_emissions_data.emissions_rate,
            "Model Architecture": model_architecture
        }
    )
