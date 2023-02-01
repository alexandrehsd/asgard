import numpy as np


def split_dataset(data, train_ratio=0.8, validation_ratio=0.1, random_state=42):
    """
    Splits a given dataset into train, validation, and test sets.

    Parameters:
    data (dataframe): The original dataset to be split
    train_ratio (float, optional): The ratio of data to be included in the train set. Default is 0.6.
    validation_ratio (float, optional): The ratio of data to be included in the validation set. Default is 0.2.
    random_state (int, optional): The seed used by the random number generator. Default is 42.

    Returns:
    tuple: Three numpy arrays corresponding to the train, validation, and test sets respectively.
    """
    data = data.sample(frac=1, random_state=random_state).reset_index(drop=True)

    train_end_index = int(train_ratio * data.shape[0])
    validation_end_index = int((train_ratio + validation_ratio) * data.shape[0])

    train = data[:train_end_index]
    validation = data[train_end_index:validation_end_index]
    test = data[validation_end_index:]

    return {"train": (np.array(train["text"]), np.array(train.iloc[:, 1:])),
            "validation": (np.array(validation["text"]), np.array(validation.iloc[:, 1:])),
            "test": (np.array(test["text"]), np.array(test.iloc[:, 1:])),
            "columns": list(data.columns[1:])
            }