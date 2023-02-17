import glob
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from asgard.utils.monitor import LOGGER


def load_datasets(csv_filepath):
    files = sorted(glob.glob(csv_filepath))

    datasets = []
    for file in files:
        datasets.append(pd.read_csv(file, sep="\t"))

    return datasets


def build_dataset(datasets, random_state=42):
    for i, dataset in enumerate(datasets):
        mlb = MultiLabelBinarizer()
        targets = mlb.fit_transform(
            dataset["Sustainable Development Goals (2021)"]
            .str.replace(" ", "")
            .str.split("|")
        )
        targets_dataframe = pd.DataFrame(targets, columns=mlb.classes_, dtype=np.int64)

        datasets[i] = pd.concat([datasets[i], targets_dataframe], axis=1)
        datasets[i] = datasets[i].drop(columns=["Sustainable Development Goals (2021)"])

    sdg_datasets = []
    for dataset in datasets:
        sdg_datasets.append(dataset)

    dataset = (
        pd.concat(sdg_datasets)
        .rename(columns={"Title": "text"})
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )

    n_targets = len(dataset.columns) - 1
    columns = ["text"] + [f"SDG{i}" for i in range(1, n_targets + 1)]

    dataset = dataset[columns]
    return dataset


def remove_duplicates(dataset):
    # remove duplicate data
    dataset = dataset.drop_duplicates()

    # perform union set on labels for duplicated text entries but different target
    # sets
    text_data = dataset[["text"]].copy()
    sdg_columns = [col for col in dataset.columns if col.startswith("SDG")]

    title_counts = text_data["text"].value_counts()
    duplicate_titles = title_counts[title_counts > 1].index.tolist()

    aggregated_rows = []
    for title in duplicate_titles:
        title_data = dataset[sdg_columns].loc[dataset["text"] == title, :]
        sdgs = title_data.sum(axis=0) > 0
        sdgs = sdgs.astype(int).tolist()

        agg_data = [title]
        agg_data.extend(sdgs)

        aggregated_rows.append(agg_data)
    deduplicate_records = pd.DataFrame(aggregated_rows, columns=["text"] + sdg_columns)

    deduplicate_dataset = dataset.loc[~dataset["text"].isin(duplicate_titles)]
    deduplicate_dataset = pd.concat(
        [deduplicate_dataset, deduplicate_records], ignore_index=True
    )

    return deduplicate_dataset


def balance_multilabel_dataset(dataset, quantile=0.5, random_state=42):
    """
    Balance the counts of target labels in a multilabel dataset.

    The function balances the counts of target labels in a dataset by iteratively sampling instances from
    classes with more instances until all classes have the same count of instances. The sample is chosen so
    that it maintains the ratio of labels within the target columns.

    Parameters:
    dataset (dataframe): The unbalanced dataset
    quantile (float, optional): the quantile of the counts of labels at which to balance the dataset.
    If you choose 1, the dataset will remain the same. If you choose 0, than the dataset will be almost
    as balanced as possible. however, you'll probably lose many data. Default value is 0.5 (median).
    random_state (int, optional): the random seed used to sample the dataframe. Default is 42.

    Returns:
    dataset (dataframe): a balanced dataset.
    Example:
    >>> unbalanced_dataset = pd.read_csv("unbalanced_dataset.csv")
    >>> balanced_dataset = balance_multilabel_dataset(unbalanced_dataset)
    """
    # compute the overall counts of labels in the dataset before multilabel balancing
    sdg_counts = dataset.iloc[:, 1:].sum(axis=0)

    # compute the quantile label count and identify those labels below it
    quantile_label_count = np.quantile(sdg_counts, q=quantile)

    # compute the number of samples to add from each label to reach the quantile of the
    # number of samples
    samples_to_add = quantile_label_count - sdg_counts
    samples_to_add[samples_to_add < 0] = quantile_label_count

    # sort label from the minimum to maximum label count
    sorted_labels = dataset.iloc[:, 1:].sum(axis=0).sort_values().index.tolist()

    balanced_dataset = pd.DataFrame(columns=dataset.columns)

    for label in sorted_labels:
        # compute the number of records that must be sampled for current label
        label_samples_to_add = int(samples_to_add[label])
        samples_available = np.sum(dataset[label] == 1)

        label_has_samples_available = samples_available > 0
        label_needs_more_samples = label_samples_to_add > 0

        if label_has_samples_available and label_needs_more_samples:
            # creates a mask to filter the dataset with the samples from current label
            samples_from_selected_label = dataset[label] == 1

            # guarantee that it will not try to add more samples than there is available
            if label_samples_to_add > samples_available:
                label_samples_to_add = samples_available

            # samples the dataset
            selected_samples = dataset[samples_from_selected_label].sample(
                n=label_samples_to_add, random_state=random_state
            )

            # remove the selected samples from the dataset in order
            # to avoid those samples in the next iteration
            dataset = dataset[~samples_from_selected_label]

            # concatenate the balanced_dataset with the selected_samples
            balanced_dataset = pd.concat([balanced_dataset, selected_samples])

        # update the counts of samples to be added to the next labels
        balanced_label_count = balanced_dataset.iloc[:, 1:].sum(axis=0)
        samples_to_add = quantile_label_count - balanced_label_count

    return balanced_dataset


def load_dataset(csv_filepath="./data/raw/sdg", balance_quantile=0.5, random_state=42):
    csv_filepath = os.path.join(csv_filepath, "*.csv")

    LOGGER.info("Loading and building datasets.")
    datasets = load_datasets(csv_filepath)
    dataset = build_dataset(datasets, random_state=random_state)

    LOGGER.info("Removing duplicate titles.")
    dataset = remove_duplicates(dataset)

    LOGGER.info("Balancing the dataset.")
    balanced_dataset = balance_multilabel_dataset(
        dataset, quantile=balance_quantile, random_state=random_state
    )

    return balanced_dataset
