import os
import pandas as pd
from asgard.dataset.loader import load_datasets, build_dataset, remove_duplicates

CSV_FILEPATH = "./tests/unit/mock_data"


def test_load_datasets():
    data = load_datasets(csv_filepath=os.path.join(CSV_FILEPATH, "raw/*.csv"))
    assert isinstance(data, list)
    assert len(data) == 2


def test_build_dataset():
    data_mock1 = pd.read_csv(os.path.join(CSV_FILEPATH, "raw/mock_data_1.csv"), sep="\t")
    data_mock2 = pd.read_csv(os.path.join(CSV_FILEPATH, "raw/mock_data_2.csv"), sep="\t")

    datasets = [data_mock1, data_mock2]
    dataset = build_dataset(datasets, random_state=42)
    assert isinstance(dataset, pd.DataFrame)
    assert dataset.shape == (40, 5)
    assert dataset.columns.tolist() == ["text", "SDG1", "SDG2", "SDG3", "SDG4"]


def test_remove_duplicates():
    dataset = pd.read_csv(os.path.join(CSV_FILEPATH, "dataset_build/mock_data_build.csv"), sep="\t")

    deduplicate_dataset = remove_duplicates(dataset)
    assert deduplicate_dataset.shape == (39, 5)

