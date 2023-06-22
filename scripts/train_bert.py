import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from official.nlp import optimization  # to create AdamW optimizer
from tensorflow import keras
from asgard.metrics.metrics import HammingScoreMetric

from asgard.metrics.metrics import (
    compute_binary_metrics,
    print_multilabel_metrics,
)
from asgard.utils.logging import get_run_logdir, log_to_wandb
from asgard.models.bert.bert import build_model
from asgard.utils.weights import get_class_weight, get_weighted_loss
from asgard.utils.data_loader import load_datasets
from asgard.utils.monitor import LOGGER

tf.get_logger().setLevel('ERROR')


# noinspection PyShadowingNames
def train_model(
        train_set,
        valid_set,
        test_set,
        class_weight_kind,
        optimizer,
        dropout,
        epochs,
        output_path,
        log=True,
):
    # bert model
    bert_model_name = "bert_en_uncased_L-12_H-768_A-12"
    tfhub_encoder_handler = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3'
    tfhub_preprocess_handler = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'

    # Loss function
    LOGGER.info("Computing weights for the loss function.")
    class_weights = get_class_weight(train_set, class_weight_kind)

    loss = None  # initialize variable
    if (class_weight_kind is None) or (class_weight_kind == "None"):
        loss = "binary_crossentropy"
    elif (class_weight_kind == "balanced") or (class_weight_kind == "two-to-one"):
        loss = get_weighted_loss(class_weights)

    LOGGER.info("Building model.")
    n_outputs = 16
    metrics = [HammingScoreMetric()]
    model = build_model(
        encoder_handler=tfhub_encoder_handler,
        preprocess_handler=tfhub_preprocess_handler,
        n_outputs=n_outputs,
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
        dropout=dropout
    )

    # Define callbacks
    root_logdir = os.path.join(output_path, "logs")
    run_logdir = get_run_logdir(root_logdir)
    model_dir = os.path.join(output_path, "artifact")

    tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
    early_stopping_cb = keras.callbacks.EarlyStopping(
        monitor="val_hamming_score", mode="max", min_delta=0.002, restore_best_weights=True
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
    # wandb.init()

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
        "--dropout",
        type=float,
        required=False,
        help="Dropout rate in the hidden and recurrent layers",
        default=0.1,
    )
    parser.add_argument(
        "-arg3",
        "--epochs",
        type=int,
        required=False,
        help="Number of epochs",
        default=4,
    )
    # learning rate arguments
    parser.add_argument(
        "-arg4",
        "--learning_rate",
        type=float,
        required=False,
        help="Initial learning rate",
        default=3e-5,
    )

    args = parser.parse_args()

    # Define number of epochs
    epochs = args.epochs
    steps_per_epoch = 24813

    # Define optimizer
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1 * num_train_steps)

    optimizer = optimization.create_optimizer(init_lr=args.learning_rate,
                                              num_train_steps=num_train_steps,
                                              num_warmup_steps=num_warmup_steps,
                                              optimizer_type='adamw')

    class_weight_kind = args.class_weight_kind
    dropout = args.dropout

    output_path = "./models/bert"

    LOGGER.info("Loading the datasets.")
    train_set, valid_set, test_set = load_datasets("storage/datasets/tf_raw")

    LOGGER.info("Starting training routine.")
    train_model(
        train_set,
        valid_set,
        test_set,
        class_weight_kind,
        optimizer,
        dropout,
        epochs,
        output_path,
        log=False,
    )
