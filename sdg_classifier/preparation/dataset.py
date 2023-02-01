# Import necessary modules
import os
import glob

import numpy as np
import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer

current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
os.chdir(parent_dir)


def load_datasets(csv_filepath):
    files = glob.glob(csv_filepath)

    datasets = []
    for file in files:
        datasets.append(pd.read_csv(file, sep="\t"))

    return datasets


def build_dataset(datasets):
    for i, dataset in enumerate(datasets):
        mlb = MultiLabelBinarizer()
        targets = mlb.fit_transform(dataset["Sustainable Development Goals (2021)"]
                                    .str.replace(" ", "")
                                    .str.split("|")
                                    )
        targets_dataframe = pd.DataFrame(targets, columns=mlb.classes_, dtype=np.int64)

        datasets[i] = pd.concat([datasets[i], targets_dataframe], axis=1)
        datasets[i] = datasets[i].drop(columns=["Sustainable Development Goals (2021)"])

    sdg_datasets = []
    for dataset in datasets:
        sdg_datasets.append(dataset)

    dataset = pd.concat(sdg_datasets)

    return dataset


def remove_duplicates(dataset):

    dataset = dataset.rename(columns={"Title": "text"}).drop_duplicates()

    # Remove duplicate titles with different targets
    counts = dataset["text"].value_counts()
    titles, counts = list(counts.index), list(counts)

    # TODO: Refactor this code to aggregate all records with the same text and different targets
    for title, count in zip(titles, counts):
        if count > 1:
            dataset = dataset.loc[dataset["text"] != title, :]

        # since the list of counts is ordered, if it gets to count == 1, then we can break the loop
        if count == 1:
            break

    dataset = dataset.reset_index(drop=True)
    return dataset


def balance_multilabel_dataset(dataset):
    titles = dataset[["text"]]
    unbalanced_targets = dataset.iloc[:, 1:]

    # TODO: Refactor this code to be independent from the sequence of files that is inputed
    # step 1
    counts = unbalanced_targets.sum(axis=0)
    base_class_count, base_class_idx = np.min(counts), np.argmin(counts)

    # step 2
    # initiliaze keeper dataset
    keeper = unbalanced_targets[unbalanced_targets.iloc[:, base_class_idx] == 1]

    # remove records added to the keeper dataset
    unbalanced_targets = unbalanced_targets[unbalanced_targets.iloc[:, base_class_idx] == 0]

    # step 3
    # identify classes from keeper that have more instances than base_class_count
    intransigent = np.sum(keeper, axis=0) >= base_class_count

    while True:

        compromised = np.sum(keeper, axis=0) < base_class_count

        # step 5.1: check if compromised stopped changing
        if np.all(intransigent == compromised):
            break

        # step 4
        # step 4.1
        intransigent_classes_idx = np.concatenate(np.argwhere(np.array(~compromised)))

        balance_mask = np.full((unbalanced_targets.shape[0],), True)
        for j in intransigent_classes_idx:
            balance_mask = balance_mask & (unbalanced_targets.iloc[:, j] == 0)

        concession = unbalanced_targets.loc[balance_mask, :]
        unbalanced_targets = unbalanced_targets.loc[balance_mask, :]

        # step 5.1: check if concession only have 0's (is empty)
        if sum(np.sum(concession)) == 0:
            break

        # step 4.2
        compromised_classes_idx = np.array(compromised).nonzero()[0]

        if len(compromised_classes_idx) > 0:
            compromised_class = np.array(compromised).nonzero()[0][0]

            n_sampleable = np.sum(concession.iloc[:, compromised_class])

            n_samples = base_class_count - np.sum(keeper, axis=0)[compromised_classes_idx[0]]

            if n_samples > n_sampleable:
                n_samples = n_sampleable

            unbalanced_targets = unbalanced_targets.loc[concession.iloc[:, compromised_class] == 0, :]
            concession = concession[concession.iloc[:, compromised_class] == 1][:n_samples]

        # update keeper and intransigent sets for the next iteration
        keeper = pd.concat([keeper, concession])
        intransigent = compromised

    balanced_targets = keeper.astype(np.float32)

    balanced_targets["index"] = balanced_targets.index
    titles = titles.assign(index=titles.index)
    balanced_dataset = (balanced_targets
                        .merge(titles, how="left", on="index")
                        .set_index("index")
                        )

    columns = ["text", "SDG1", "SDG2", "SDG3", "SDG4", "SDG5", "SDG6", "SDG7", "SDG8", "SDG9",
               "SDG10", "SDG11", "SDG12", "SDG13", "SDG14", "SDG15", "SDG16"]
    balanced_dataset = balanced_dataset[columns]

    return balanced_dataset


def load_dataset(csv_filepath="./data/csv/sdg"):
    csv_filepath = os.path.join(csv_filepath, "*.csv")

    datasets = load_datasets(csv_filepath)
    dataset = build_dataset(datasets)
    dataset = remove_duplicates(dataset)
    dataset = balance_multilabel_dataset(dataset)

    return dataset

# import pandas as pd
# from sklearn.utils import resample
#
# def balance_dataset(df):
#     # Create a list of the target label columns
#     label_cols = [col for col in df.columns if col != 'text']
#
#     # Split the dataframe into separate dataframes for each target label
#     label_dfs = [df[df[col] == 1] for col in label_cols]
#
#     # Get the size of the largest dataframe
#     max_size = max([len(label_df) for label_df in label_dfs])
#
#     # Resample each dataframe to have the same size as the largest dataframe
#     resampled_dfs = [resample(label_df, replace=True, n_samples=max_size, random_state=42) for label_df in label_dfs]
#
#     # Concatenate the resampled dataframes into a single dataframe
#     balanced_df = pd.concat(resampled_dfs)
#
#     return balanced_df
