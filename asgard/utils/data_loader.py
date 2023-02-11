import os
import tensorflow as tf


# load pre-saved tensorflow datasets
def load_datasets(datapath):

    train_set = tf.data.experimental.load(os.path.join(datapath, "train_set"))
    valid_set = tf.data.experimental.load(os.path.join(datapath, "valid_set"))
    test_set = tf.data.experimental.load(os.path.join(datapath, "test_set"))

    return train_set, valid_set, test_set
