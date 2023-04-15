import os
import yaml
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import wandb
import multiprocessing
from tensorflow import keras

from asgard.metrics.metrics import (
    compute_binary_metrics,
    print_multilabel_metrics,
)
from asgard.models.rnn.logging import get_run_logdir, log_to_wandb
from asgard.models.rnn.rnn import build_model, create_text_vectorization_layer
from asgard.models.rnn.weights import get_class_weight, get_weighted_loss
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
    if class_weight_kind is None:
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
    n_outputs = 16
    model = build_model(
        n_outputs,
        text_vectorization_layer,
        optimizer,
        loss,
        units,
        dropout,
        constraint,
        n_hidden,
    )

    # Define callbacks
    root_logdir = os.path.join(output_path, "logs")
    run_logdir = get_run_logdir(root_logdir)
    model_dir = os.path.join(output_path, "artifact")

    tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
    early_stopping_cb = keras.callbacks.EarlyStopping(
        monitor="val_accuracy", mode="max", patience=2, restore_best_weights=True
    )

    callbacks = [tensorboard_cb, early_stopping_cb]

    LOGGER.info("Fitting the model.")
    history = model.fit(
        train_set, validation_data=valid_set, epochs=epochs, callbacks=callbacks
    )

    # Logging metrics
    if log:
        log_to_wandb(model, valid_set, test_set)
    else:
        y_pred = (model.predict(test_set) > 0.5) + 0
        y_true = np.concatenate([y.numpy() for X, y in test_set])
        labels = [f"SDG{i+1}" for i in range(16)]

        binary_metrics = compute_binary_metrics(y_true, y_pred, labels)

        print(binary_metrics)
        print_multilabel_metrics(y_true, y_pred)

    # save model
    model.save(model_dir)

    return model, history


def sweep_train(config_defaults=None):

    wandb.init(config=config_defaults)
    wandb.config.architecture_name = "RNN"
    wandb.config.dataset_name = "SDG-titles"

    LOGGER.info("Loading the datasets.")
    train_set, valid_set, test_set = load_datasets("storage/datasets/tf")

    config = wandb.config

    # learning rate definition
    initial_learning_rate = config.learning_rate
    decay_steps = 24813  # number of steps per epoch
    rate = 2
    decay_rate = (
        1 / rate
    )  # decrease the learning by a factor of 'rate' every 'decay_steps'
    lr_scheduler = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps, decay_rate
    )

    learning_rate = lr_scheduler

    # weight constraint definition
    constraint = keras.constraints.MaxNorm(max_value=config.weight_constraint)
    class_weight_kind = config.class_weight_kind
    output_sequence_length = config.output_sequence_length
    optimizer = config.optimizer
    units = config.units
    dropout = config.dropout
    n_hidden = config.n_hidden
    epochs = config.epochs
    output_path = "./models"

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
    )


def sweep_job():
    with open("./asgard/configs/wandb/sweeps/rnn.yaml", "r") as sweep_yaml:
        sweep_config = yaml.safe_load(sweep_yaml)

    sweep_id = wandb.sweep(sweep_config, project="ASGARD-RNN")

    n_runs = 15  # each cpu runs 15 times
    wandb.agent(sweep_id, function=sweep_train, count=n_runs)


if __name__ == "__main__":
    n_jobs = multiprocessing.cpu_count()

    # Create a pool of 2 processes
    pool = multiprocessing.Pool(processes=n_jobs)

    # Start one instance of the script on each CPU
    results = [pool.apply_async(sweep_job) for i in range(n_jobs)]

    # Wait for all processes to complete
    pool.close()
    pool.join()
