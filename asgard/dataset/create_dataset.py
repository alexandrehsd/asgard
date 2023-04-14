import argparse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # noqa E402

import pprint
import random

import numpy as np
import tensorflow as tf
from toolbox.loader import load_dataset
from toolbox.model_selection import split_dataset
from toolbox.preprocessing import preprocess_data

from asgard.utils.monitor import LOGGER


def create_dataset(X, y):
    return tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))


def log_label_counts(y_train, y_valid, y_test, labels):
    # Count labels for each data set
    train_counts = y_train.sum(axis=0)
    train_label_counts = {labels[i]: train_counts[i] for i in range(len(labels))}

    pprint.pprint(
        f"The Train set has {y_train.shape[0]} records. Label counts:\n"
        f"{train_label_counts}"
    )

    valid_counts = y_valid.sum(axis=0)
    valid_label_counts = {labels[i]: valid_counts[i] for i in range(len(labels))}
    pprint.pprint(
        f"The Validation set has {y_valid.shape[0]} records. Label counts:\n"
        f"{valid_label_counts}"
    )

    test_counts = y_test.sum(axis=0)
    test_label_counts = {labels[i]: test_counts[i] for i in range(len(labels))}

    pprint.pprint(
        f"The Test set has {y_test.shape[0]} records. Label counts:\n"
        f"{test_label_counts}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-arg1",
        "--input_csv_filepath",
        type=str,
        required=True,
        help="Input path to the raw CSV data",
    )
    parser.add_argument(
        "-arg2",
        "--output_tf_filepath",
        type=str,
        required=True,
        help="Output path to the Tensorflow data",
    )
    parser.add_argument(
        "-arg3", "--seed", type=int, required=False, help="Random seed", default=42
    )
    parser.add_argument(
        "-arg4",
        "--quantile",
        type=float,
        required=False,
        help="Dataset balance quantile",
        default=0.5,
    )
    parser.add_argument(
        "-arg5",
        "--train_ratio",
        type=float,
        required=False,
        help="Train ratio size",
        default=0.8,
    )
    parser.add_argument(
        "-arg6",
        "--validation_ratio",
        type=float,
        required=False,
        help="Validation ratio size",
        default=0.1,
    )
    parser.add_argument(
        "-arg7",
        "--text_truncation",
        type=str,
        required=False,
        help="Text standardization method (available options are 'lemma', 'stem', or None)",
        default="lemma",
    )
    parser.add_argument(
        "-arg8",
        "--batch_size",
        type=int,
        required=False,
        help="Tensorflow batch size",
        default=32,
    )

    args = parser.parse_args()
    random_state = args.seed

    np.random.seed(random_state)
    random.seed(random_state)

    dataset = load_dataset(args.input_csv_filepath, args.quantile, random_state)

    train_ratio = args.train_ratio
    validation_ratio = args.validation_ratio
    if train_ratio + validation_ratio > 1:
        raise ValueError(
            "The sum of the train_ratio and the validation_ratio cannot be greater than 1."
        )

    data_split = split_dataset(
        dataset,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
        random_state=random_state,
    )

    X_train, y_train = data_split["train"]
    X_valid, y_valid = data_split["validation"]
    X_test, y_test = data_split["test"]
    labels = data_split["labels"]

    truncation = args.text_truncation

    LOGGER.info("Preprocessing the training set titles.")
    X_train, y_train = preprocess_data(X_train, y_train, truncation=truncation)

    LOGGER.info("Preprocessing the validation set titles.")
    X_valid, y_valid = preprocess_data(X_valid, y_valid, truncation=truncation)

    LOGGER.info("Preprocessing the test set titles.")
    X_test, y_test = preprocess_data(X_test, y_test, truncation=truncation)

    log_label_counts(y_train, y_valid, y_test, labels)

    batch_size = args.batch_size

    # build train set
    train_set = (
        create_dataset(X_train, y_train)
        .shuffle(X_train.shape[0], seed=random_state)
        .batch(batch_size)
        .prefetch(1)
    )

    # build validation set
    valid_set = create_dataset(X_valid, y_valid).batch(batch_size).prefetch(1)

    # build test set
    test_set = create_dataset(X_test, y_test).batch(batch_size).prefetch(1)

    output_path = args.output_tf_filepath

    # stores tf datasets
    LOGGER.info("Storing the output datasets.")
    tf.data.experimental.save(train_set, os.path.join(output_path, "train_set"))
    tf.data.experimental.save(valid_set, os.path.join(output_path, "valid_set"))
    tf.data.experimental.save(test_set, os.path.join(output_path, "test_set"))


if __name__ == "__main__":
    main()
