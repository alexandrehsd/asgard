# Import necessary modules
import os
import argparse
import random
import logging

import numpy as np
import tensorflow as tf

from dataset import load_dataset
from preprocessing import preprocess_data
from model_selection import split_dataset

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)


def create_dataset(X, y):
    return tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-arg1", "--input_csv_filepath", type=str, required=True,
                        help="Input path to the raw CSV data")
    parser.add_argument("-arg2", "--output_tf_filepath", type=str, required=True,
                        help="Output path to the Tensorflow data")
    parser.add_argument("-arg3", "--seed", type=int, required=False,
                        help="Random seed", default=42)
    parser.add_argument("-arg4", "--quantile", type=float, required=False,
                        help="Dataset balance quantile", default=0.5)
    parser.add_argument("-arg5", "--train_ratio", type=float, required=False,
                        help="Train ratio size", default=0.8)
    parser.add_argument("-arg6", "--validation_ratio", type=float, required=False,
                        help="Validation ratio size", default=0.1)
    parser.add_argument("-arg7", "--text_truncation", type=str, required=False,
                        help="Text standardization method (available options are 'lemma', 'stem', or None)",
                        default="lemma")
    parser.add_argument("-arg8", "--batch_size", type=int, required=False,
                        help="Tensorflow batch size", default=32)

    args = parser.parse_args()
    random_state = args.seed

    np.random.seed(random_state)
    random.seed(random_state)

    dataset = load_dataset(args.input_csv_filepath, args.quantile, random_state)

    train_ratio = args.train_ratio
    validation_ratio = args.validation_ratio
    if train_ratio + validation_ratio > 1:
        raise ValueError("The sum of the train_ratio and the validation_ratio cannot be greater than 1.")

    data_split = split_dataset(dataset, train_ratio=train_ratio, validation_ratio=validation_ratio,
                               random_state=random_state)

    X_train, y_train = data_split["train"]
    X_valid, y_valid = data_split["validation"]
    X_test, y_test = data_split["test"]

    logging.info(f"The Train set has {X_train.shape[0]} records.")
    logging.info(f"The Validation set has {X_valid.shape[0]} records.")
    logging.info(f"The Test set has {X_test.shape[0]} records.")

    truncation = args.text_truncation

    logging.info("Preprocessing the training set titles.")
    X_train, y_train = preprocess_data(X_train, y_train, truncation=truncation)

    logging.info("Preprocessing the validation set titles.")
    X_valid, y_valid = preprocess_data(X_valid, y_valid, truncation=truncation)

    logging.info("Preprocessing the test set titles.")
    X_test, y_test = preprocess_data(X_test, y_test, truncation=truncation)

    batch_size = args.batch_size

    # build train set
    train_set = create_dataset(X_train, y_train). \
        shuffle(X_train.shape[0], seed=random_state).batch(batch_size).prefetch(1)

    # build validation set
    valid_set = create_dataset(X_valid, y_valid).batch(batch_size).prefetch(1)

    # build test set
    test_set = create_dataset(X_test, y_test).batch(batch_size).prefetch(1)

    output_path = args.output_tf_filepath

    # stores tf datasets
    logging.info("Storing the output datasets.")
    tf.data.experimental.save(train_set, os.path.join(output_path, "train_set"))
    tf.data.experimental.save(valid_set, os.path.join(output_path, "valid_set"))
    tf.data.experimental.save(test_set, os.path.join(output_path, "test_set"))


if __name__ == "__main__":
    main()
