import os
import pandas as pd
from asgard.dataset.model_selection import split_dataset


CSV_FILEPATH = "./tests/unit/test_data"


def test_split_dataset():
    datapath = os.path.join(CSV_FILEPATH, "dataset_build/test_data_build.csv")
    dataset = pd.read_csv(datapath, sep="\t")

    data_split = split_dataset(dataset)

    assert list(data_split.keys()) == ['train', 'validation', 'test', 'labels']
    assert data_split["labels"] == dataset.columns[1:].tolist()
    assert data_split["train"][1].shape[1] == len(dataset.columns[1:])
