import numpy as np
import tensorflow as tf
from tensorflow import keras
from asgard.metrics.metrics import HammingScoreMetric


def create_text_vectorization_layer(train_set, output_sequence_length):
    # convert tensorflow train set to numpy train set
    X_train = np.concatenate([data.numpy() for data, label in train_set])

    max_vocabulary_size = 20000  # recommend to be 10000+

    text_vectorization = tf.keras.layers.TextVectorization(
        max_tokens=max_vocabulary_size,
        output_mode="int",
        output_sequence_length=output_sequence_length,
    )

    # "adapt" the layer to the data (this is the same as "fit")
    text_vectorization.adapt(X_train)

    return text_vectorization


def build_model(
    n_outputs,
    text_vectorization,
    optimizer,
    loss,
    units=50,
    dropout=0.25,
    constraint=None,
    n_hidden=2,
):
    # add 1 for the padding token
    vocabulary_size = len(text_vectorization.get_vocabulary()) + 1
    number_out_of_vocabulary_buckets = (
        1  # default value for the text vectorizarization layer, do not change
    )

    # set embedding dimensions to the number of units
    embed_size = units

    # same droupout and constraint rate for the recurrent states
    recurrent_dropout = dropout  # must be set to 0 when using GPU
    recurrent_constraint = constraint

    # instantiate the model and add text vectorization and embedding layer
    model = keras.models.Sequential()
    model.add(text_vectorization)
    model.add(
        keras.layers.Embedding(
            input_dim=vocabulary_size + number_out_of_vocabulary_buckets,
            output_dim=embed_size,
            mask_zero=True,
            input_shape=[None],
        )
    )

    # add hidden layers
    for layer in range(n_hidden - 1):
        model.add(
            keras.layers.GRU(
                units,
                return_sequences=True,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
                kernel_constraint=constraint,
                recurrent_constraint=recurrent_constraint,
            )
        )

    model.add(
        keras.layers.GRU(
            units,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            kernel_constraint=constraint,
            recurrent_constraint=recurrent_constraint,
        )
    )

    # add output layer
    model.add(keras.layers.Dense(n_outputs, activation="sigmoid"))

    model.compile(loss=loss, optimizer=optimizer, metrics=[HammingScoreMetric()])
    return model
