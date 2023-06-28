import tensorflow as tf
from tensorflow import keras
from asgard.utils.monitor import LOGGER


class EarlyStoppingHammingScore(keras.callbacks.Callback):
    """Stop training when the Hamming Score (HS) is at its max, i.e. the HS stops increasing.
    Arguments:
    patience: Number of epochs to wait after min has been hit. After this number of no improvement, training stops.
    """

    def __init__(self, monitor="val_hamming_score", patience=0, min_delta=0, restore_best_weights=True):
        super().__init__()
        self.min_delta = min_delta
        self.patience = patience
        self.monitor = monitor
        self.restore_best_weights = restore_best_weights
        
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when HS is no longer maximum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as -infinity.
        self.best = tf.constant(-float("inf"))

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current > (self.best + self.min_delta):
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                # Record the best weights if current results is better (greater).
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights:
                    self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            LOGGER.info("Epoch %03d: Early Stopping" % (self.stopped_epoch + 1))