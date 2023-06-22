import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras


def compute_class_weights(train_set):
    i = 0
    for X, y in train_set:
        if i == 0:
            y_train = y.numpy()
            i += 1

        y_train = np.concatenate((y_train, y.numpy()))

    n_classes = y_train.shape[1]
    weights = np.empty([n_classes, 2])
    for i in range(n_classes):
        weights[i] = compute_class_weight(
            "balanced", classes=[0.0, 1.0], y=y_train[:, i]
        )
    return weights


# Weighted binary cross entropy loss function
# use custom loss function
# https://stackoverflow.com/questions/48485870/multi-label-classification-with-class-weights-in-keras
def get_weighted_loss(weights):
    def weighted_loss(y_train, y_pred):
        return keras.backend.mean(
            (weights[:, 0] ** (1 - y_train))
            * (weights[:, 1] ** y_train)
            * keras.backend.binary_crossentropy(y_train, y_pred),
            axis=-1,
        )

    return weighted_loss


def get_class_weight(train_set, class_weight_kind="balanced"):
    if (class_weight_kind is None) or (class_weight_kind == "None"):
        class_weights = None

    elif class_weight_kind == "balanced":
        class_weights = compute_class_weights(train_set)

    elif class_weight_kind == "two-to-one":
        class_weights = np.zeros((16, 2))
        class_weights[:, 0] = 1.0
        class_weights[:, 1] = 2.0

    return class_weights
