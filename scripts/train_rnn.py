import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import wandb
from tensorflow import keras
# tf.config.run_functions_eagerly(True)

from codecarbon import EmissionsTracker
from asgard.metrics.metrics import HammingScoreMetric
from asgard.callbacks.callbacks import EarlyStoppingHammingScore
from asgard.metrics.metrics import (
    compute_binary_metrics,
    print_multilabel_metrics,
)
from asgard.utils.logging import get_run_id, log_to_wandb
from asgard.models.rnn.rnn import build_model, create_text_vectorization_layer
from asgard.utils.weights import get_class_weight, get_weighted_loss
from asgard.utils.data_loader import load_datasets
from asgard.utils.monitor import LOGGER


# noinspection PyShadowingNames
def train_model(
    train_set,
    valid_set,
    test_set,
    class_weight_kind,
    optimizer,
    learning_rate,
    units,
    dropout,
    constraint,
    n_hidden,
    output_sequence_length,
    epochs,
    output_path,
    model_architecture,
    log=True,
):
    # train preparation
    LOGGER.info("Creating the text vectorization layer.")
    text_vectorization_layer = create_text_vectorization_layer(
        train_set, output_sequence_length
    )

    # Loss function
    LOGGER.info("Computing weights for the loss function.")
    class_weights = get_class_weight(train_set, class_weight_kind)

    loss = None  # initialize variable
    if (class_weight_kind is None) or (class_weight_kind == "None"):
        loss = "binary_crossentropy"
    elif (class_weight_kind == "balanced") or (class_weight_kind == "two-to-one"):
        loss = get_weighted_loss(class_weights)

    # Optimizer
    if optimizer == "Nadam":
        optimizer = keras.optimizers.Nadam(learning_rate, clipnorm=1)
    elif optimizer == "Adam":
        optimizer = keras.optimizers.Adam(learning_rate, clipnorm=1)
    elif optimizer == "RMSprop":
        optimizer = keras.optimizers.RMSprop(learning_rate, clipnorm=1, centered=True)

    LOGGER.info("Building model.")
    emissions_tracker = EmissionsTracker(log_level="critical")
    emissions_tracker.start()
    
    n_outputs = 16
    metrics = [HammingScoreMetric()]
    model = build_model(
        n_outputs,
        text_vectorization_layer,
        optimizer,
        loss,
        units,
        dropout,
        constraint,
        n_hidden,
        metrics=metrics
    )

    # Define callbacks
    run_id = get_run_id()
    run_logdir = os.path.join(output_path, "rnn", "logs", run_id)

    tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
    early_stopping_cb = EarlyStoppingHammingScore(monitor="val_hamming_score", patience=1, min_delta=0.002)
    wandb_cb = wandb.keras.WandbCallback(monitor="val_hamming_score", mode="max")
    
    callbacks = [tensorboard_cb, early_stopping_cb, wandb_cb]

    LOGGER.info("Fitting the model.")
    history = model.fit(
        train_set, validation_data=valid_set, epochs=epochs, callbacks=callbacks
    )
    
    emissions_tracker.stop()

    # Logging metrics
    if log:
        log_to_wandb(model, valid_set, test_set, emissions_tracker, model_architecture)
        wandb.save(os.path.join(wandb.run.dir, "model"))
    else:
        y_pred = (model.predict(test_set) > 0.5) + 0
        y_true = np.concatenate([y.numpy() for X, y in test_set])
        labels = [f"SDG{i+1}" for i in range(16)]

        binary_metrics = compute_binary_metrics(y_true, y_pred, labels)

        print(binary_metrics)
        print_multilabel_metrics(y_true, y_pred)

    return model, history


if __name__ == "__main__":
    wandb.init()

    parser = argparse.ArgumentParser()
    # model arguments
    parser.add_argument(
        "-arg1",
        "--class_weight_kind",
        type=str,
        required=False,
        help="Class weighting strategy",
        default=None,
    )
    parser.add_argument(
        "-arg2",
        "--optimizer",
        type=str,
        required=False,
        help="Optimizer",
        default="RMSprop",
    )
    parser.add_argument(
        "-arg3", "--units", type=int, required=False, help="Units", default=70
    )
    parser.add_argument(
        "-arg4",
        "--dropout",
        type=float,
        required=False,
        help="Dropout rate in the hidden and recurrent layers",
        default=0.3,
    )
    parser.add_argument(
        "-arg5",
        "--weight_constraint",
        type=float,
        required=False,
        help="max value used by the MaxNorm weight constraint",
        default=2.0,
    )
    parser.add_argument(
        "-arg6",
        "--n_hidden",
        type=int,
        required=False,
        help="number of hidden layers",
        default=1,
    )
    parser.add_argument(
        "-arg7",
        "--output_sequence_length",
        type=int,
        required=False,
        help="Standard output sequence length",
        default=70,
    )
    parser.add_argument(
        "-arg8",
        "--epochs",
        type=int,
        required=False,
        help="Number of epochs",
        default=5,
    )
    # learning rate arguments
    parser.add_argument(
        "-arg9",
        "--learning_rate",
        type=float,
        required=False,
        help="Initial learning rate",
        default=0.01,
    )
    parser.add_argument(
        "-arg10",
        "--rate",
        type=float,
        required=False,
        help="Rate of the learning rate decay",
        default=2.0,
    )
    parser.add_argument(
        "-arg11",
        "--model_architecture",
        type=str,
        required=True,
        help="Model Architecture",
        default="rnn",
    )
    args = parser.parse_args()

    # learning rate definition
    initial_learning_rate = args.learning_rate
    
    LOGGER.info("Loading the datasets.")
    train_set, valid_set, test_set = load_datasets("storage/datasets/tf")
    
    num_batches = 0 
    for _ in iter(train_set):
        num_batches += 1
    
    decay_steps = num_batches  # number of steps per epoch
    rate = args.rate
    decay_rate = (
        1 / rate
    )  # decrease the learning by a factor of 'rate' every 'decay_steps'
    lr_scheduler = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps, decay_rate
    )

    learning_rate = lr_scheduler

    # weight constraint definition
    if args.weight_constraint == -1:
        weight_constraint = None
    else:
        weight_constraint = args.weight_constraint

    constraint = keras.constraints.MaxNorm(max_value=weight_constraint)
    class_weight_kind = args.class_weight_kind
    output_sequence_length = args.output_sequence_length
    optimizer = args.optimizer
    units = args.units
    dropout = args.dropout
    n_hidden = args.n_hidden
    epochs = args.epochs
    model_architecture = args.model_architecture
    output_path = "./training_runs"

    LOGGER.info("Starting training routine.")
    train_model(
        train_set,
        valid_set,
        test_set,
        class_weight_kind,
        optimizer,
        learning_rate,
        units,
        dropout,
        constraint,
        n_hidden,
        output_sequence_length,
        epochs,
        output_path,
        model_architecture
    )
