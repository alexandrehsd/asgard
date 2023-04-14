import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # noqa E402

import numpy as np
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
    log=False,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-arg1",
        "--data_path",
        type=str,
        required=True,
        help="Input path to the tensorflow datasets",
    )
    parser.add_argument(
        "-arg2",
        "--output_path",
        type=str,
        required=True,
        help="Output path to the model artifact",
    )
    # model arguments
    parser.add_argument(
        "-arg3",
        "--class_weight_kind",
        type=str,
        required=False,
        help="Class weighting strategy",
        default=None,
    )
    parser.add_argument(
        "-arg4",
        "--optimizer",
        type=str,
        required=False,
        help="Optimizer",
        default="RMSprop",
    )
    parser.add_argument(
        "-arg5", "--units", type=int, required=False, help="Units", default=70
    )
    parser.add_argument(
        "-arg6",
        "--dropout",
        type=float,
        required=False,
        help="Dropout rate in the hidden and recurrent layers",
        default=0.3,
    )
    parser.add_argument(
        "-arg7",
        "--constraint",
        type=float,
        required=False,
        help="max value used by the MaxNorm weight constraint",
        default=2.0,
    )
    parser.add_argument(
        "-arg8",
        "--n_hidden",
        type=int,
        required=False,
        help="number of hidden layers",
        default=1,
    )
    parser.add_argument(
        "-arg9",
        "--output_sequence_length",
        type=int,
        required=False,
        help="Standard output sequence length",
        default=70,
    )
    parser.add_argument(
        "-arg10",
        "--epochs",
        type=int,
        required=False,
        help="Number of epochs",
        default=5,
    )

    # learning rate arguments
    parser.add_argument(
        "-arg11",
        "--initial_lr",
        type=float,
        required=False,
        help="Initial learning rate",
        default=0.01,
    )
    parser.add_argument(
        "-arg12",
        "--decay_steps",
        type=int,
        required=False,
        help="Amount of steps to the learning rate decay",
        default=24813,
    )  # default is the number of steps per epoch
    parser.add_argument(
        "-arg13",
        "--rate",
        type=float,
        required=False,
        help="Rate of the learning rate decay",
        default=2.0,
    )
    args = parser.parse_args()

    # learning rate definition
    initial_learning_rate = args.initial_lr
    decay_steps = args.decay_steps  # number of steps per epoch
    rate = args.rate
    decay_rate = (
        1 / rate
    )  # decrease the learning by a factor of 'rate' every 'decay_steps'
    lr_scheduler = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps, decay_rate
    )

    learning_rate = lr_scheduler

    # weight constraint definition
    constraint = keras.constraints.MaxNorm(max_value=args.constraint)
    class_weight_kind = args.class_weight_kind
    output_sequence_length = args.output_sequence_length
    optimizer = args.optimizer
    units = args.units
    dropout = args.dropout
    n_hidden = args.n_hidden
    epochs = args.epochs

    LOGGER.info("Loading the datasets.")
    train_set, valid_set, test_set = load_datasets(args.data_path)

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
        args.output_path,
    )
