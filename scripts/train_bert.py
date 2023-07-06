import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
import wandb

from codecarbon import EmissionsTracker
from official.nlp import optimization  # to create AdamW optimizer
from tensorflow import keras
from asgard.callbacks.callbacks import EarlyStoppingHammingScore
from asgard.metrics.metrics import HammingScoreMetric
from asgard.metrics.metrics import (
    compute_binary_metrics,
    print_multilabel_metrics,
)
from asgard.utils.logging import get_run_id, log_to_wandb
from asgard.models.bert.bert import build_model
from asgard.utils.weights import get_class_weight, get_weighted_loss
from asgard.utils.data_loader import load_datasets
from asgard.utils.monitor import LOGGER

from absl import logging
logging.set_verbosity(logging.ERROR)

tf.get_logger().setLevel('ERROR')


def train_model(
        train_set,
        valid_set,
        test_set,
        class_weight_kind,
        optimizer,
        dropout,
        epochs,
        output_path,
        model_architecture,
        log=True,
):
    # bert model
    if model_architecture == "bert":
        # bert_model_name = "bert_en_uncased_L-12_H-768_A-12"  # noqa: F841
        tfhub_encoder_handler = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3'
        tfhub_preprocess_handler = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
        
    # distilbert
    if model_architecture == "distilbert":
        # https://tfhub.dev/jeongukjae/distilbert_en_uncased_L-6_H-768_A-12/1
        # bert_model_name = "distilbert_en_uncased_L-6_H-768_A-12"
        tfhub_encoder_handler = "https://tfhub.dev/jeongukjae/distilbert_en_uncased_L-6_H-768_A-12/1"
        tfhub_preprocess_handler = "https://tfhub.dev/jeongukjae/distilbert_en_uncased_preprocess/2"

    # Loss function
    LOGGER.info("Computing weights for the loss function.")
    class_weights = get_class_weight(train_set, class_weight_kind)

    loss = None  # initialize variable
    if (class_weight_kind is None) or (class_weight_kind == "None"):
        loss = "binary_crossentropy"
    elif (class_weight_kind == "balanced") or (class_weight_kind == "two-to-one"):
        loss = get_weighted_loss(class_weights)

    LOGGER.info("Building model.")
    emissions_tracker = EmissionsTracker(log_level="critical")
    emissions_tracker.start()
    
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
    run_id = get_run_id()
    run_logdir = os.path.join(output_path, model_architecture, "logs", run_id)

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
    parser.add_argument(
        "-arg5",
        "--model_architecture",
        type=str,
        required=True,
        help="Model Architecture",
        default="bert",
    )
    
    LOGGER.info("Loading the datasets.")
    train_set, valid_set, test_set = load_datasets("storage/datasets/tf_raw")

    args = parser.parse_args()

    # Define number of epochs
    epochs = args.epochs
    
    num_batches = 0 
    for _ in iter(train_set):
        num_batches += 1
    
    steps_per_epoch = num_batches

    # Define optimizer
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1 * num_train_steps)

    optimizer = optimization.create_optimizer(init_lr=args.learning_rate,
                                              num_train_steps=num_train_steps,
                                              num_warmup_steps=num_warmup_steps,
                                              optimizer_type='adamw')

    class_weight_kind = args.class_weight_kind
    dropout = args.dropout
    
    model_architecture = args.model_architecture
    output_path = "./training_runs"
    
    # # Define the size of the subsample
    # subsample_size = 10

    # # Shuffle the dataset
    # train_shuffled_dataset = train_set.shuffle(buffer_size=10)

    # # Take a subsample from the shuffled dataset
    # train_subsample = train_shuffled_dataset.take(subsample_size)
    
    # # Shuffle the dataset
    # valid_shuffled_dataset = valid_set.shuffle(buffer_size=10)

    # # Take a subsample from the shuffled dataset
    # valid_subsample = valid_shuffled_dataset.take(subsample_size)
    
    # # Shuffle the dataset
    # test_shuffled_dataset = test_set.shuffle(buffer_size=10)

    # # Take a subsample from the shuffled dataset
    # test_subsample = test_shuffled_dataset.take(subsample_size)

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
        model_architecture,
        log=True,
    )
