# Import necessary modules
import os
import glob

import numpy as np
import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer
from langdetect import detect

current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
os.chdir(parent_dir)

from sdg_classifier.utils.monitor import LOGGER


def load_datasets(csv_filepath):
    files = sorted(glob.glob(csv_filepath))

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
    # remove duplicate data
    dataset = dataset.drop_duplicates()

    # perform union set on labels for duplicated text entries but different target
    # sets
    text_data = dataset[["text"]].copy()
    SDG_columns = [col for col in dataset.columns if col.startswith('SDG')]

    duplicated_titles = text_data["text"].value_counts()
    duplicated_titles = duplicated_titles[duplicated_titles > 1]
    titles = list(duplicated_titles.index)

    aggregated_rows = []
    for title in titles:
        title_data = dataset[SDG_columns].loc[dataset["text"] == title, :]
        sdgs = title_data.sum(axis=0) > 0
        sdgs = sdgs.astype(int).tolist()

        agg_data = [title]
        agg_data.extend(sdgs)

        aggregated_rows.append(agg_data)
    deduplicated_records = pd.DataFrame(aggregated_rows, columns=["text"] + SDG_columns)

    dataset = (dataset
               .loc[~dataset["text"].isin(titles)]
               .append(deduplicated_records, ignore_index=True)
               )

    return dataset


def is_english(text):
    try:
        language = detect(text)
        return language == 'en'
    except:
        return False


def filter_english_text(dataset):
    is_en = [is_english(text) for text in dataset["text"]]
    return dataset[is_en]


def balance_multilabel_dataset(dataset):
    """
    Balance the counts of target labels in a multilabel dataset.

    The function balances the counts of target labels in a dataset by iteratively sampling instances from
    classes with more instances until all classes have the same count of instances. The sample is chosen so
    that it maintains the ratio of labels within the target columns.

    The steps of the process are:

    1. Identifying the class with the lowest number of instances and the count of that class.
    2. Initializing a new dataset with instances that belong to the base class.
    3. Identifying classes that have more instances than the base class count, and marking them as "intransigent."
    4. Adding instances from the other classes to the new dataset, until they have a count that is at most equal to the base class count.
    5. Repeat the previous step until either the compromised classes have stabilized, or there are no more instances to add.
    6. Finally, merging the titles with the balanced targets and returning the final balanced dataset.

    Args:
    - dataset (pd.DataFrame): The input dataset with target labels. The dataset must have two columns:
      "text" and multiple binary target columns (e.g., "SDG1", "SDG2", etc.).

    Returns:
    - pd.DataFrame: The balanced dataset with the same structure as the input, but with balanced target label counts.

    Example:
    >>> unbalanced_dataset = pd.read_csv("unbalanced_dataset.csv")
    >>> balanced_dataset = balance_multilabel_dataset(unbalanced_dataset)
    """
    titles = dataset[["text"]]

    SDG_columns = [col for col in dataset.columns if col.startswith('SDG')]
    target_columns = dataset[SDG_columns]

    # TODO: Refactor this code to be independent from the sequence of files that is inputed
    # Step 1: Find the class with the minimum number of instances
    class_counts = target_columns.sum(axis=0)
    minimum_count, base_class_index = np.min(class_counts), np.argmin(class_counts)

    # Step 2: Initialize the "keeper" dataset with all instances of the minimum class
    initial_keeper = target_columns[target_columns.iloc[:, base_class_index] == 1]
    remaining_targets = target_columns[target_columns.iloc[:, base_class_index] == 0]

    # Step 3: Keep track of which classes still have more instances than minimum_class_count
    classes_with_minimum_count = np.sum(initial_keeper, axis=0) >= minimum_count

    while True:
        # Step 4: Identify classes that have been reduced to the minimum class count
        classes_with_less_count = np.sum(initial_keeper, axis=0) < minimum_count

        # Step 5: If no classes have changed, break the loop
        if np.all(classes_with_minimum_count == classes_with_less_count):
            break

        # Step 4.1: Classes that are still over the minimum count
        classes_with_minimum_count_indexes = np.concatenate(np.argwhere(np.array(~classes_with_less_count)))

        # Step 4.2: Instances of the target_columns that don't belong to any class in
        # `classes_with_minimum_count_indexes`
        balance_mask = np.full((remaining_targets.shape[0],), True)
        for index in classes_with_minimum_count_indexes:
            balance_mask = balance_mask & (remaining_targets.iloc[:, index] == 0)

        targets_to_add = remaining_targets.loc[balance_mask, :]
        remaining_targets = remaining_targets.loc[balance_mask, :]

        # Step 5.1: If there are no instances left in targets_to_add, break the loop
        if sum(np.sum(targets_to_add)) == 0:
            break

        # Step 4.3: Identify the first class that have less indexes than the minimum
        classes_with_less_count_indexes = np.array(classes_with_less_count).nonzero()[0]

        if len(classes_with_less_count_indexes) > 0:
            class_to_add_to = np.array(classes_with_less_count).nonzero()[0][0]

            # Step 4.4: Determine the number of instances in "targets_to_add" that belong to the
            # classes_with_less_count set
            sampleable_count = np.sum(targets_to_add.iloc[:, class_to_add_to])

            # Step 4.5: Determine the number of instances needed to balance the classes_with_less_count set in
            # "initial_keeper"
            samples_needed = minimum_count - np.sum(initial_keeper, axis=0)[classes_with_less_count_indexes[0]]

            # Take the maximum number of instances from targets_to_add
            if samples_needed > sampleable_count:
                samples_needed = sampleable_count

            # Step 4.6: Limit "sample_needed" to the number of available sampleable instances
            remaining_targets = remaining_targets.loc[targets_to_add.iloc[:, class_to_add_to] == 0, :]
            targets_to_add = targets_to_add[targets_to_add.iloc[:, class_to_add_to] == 1][:samples_needed]

        # Update initial_keeper and classes_with_minimum_count sets for the next iteration
        initial_keeper = pd.concat([initial_keeper, targets_to_add])
        classes_with_minimum_count = classes_with_less_count

    balanced_targets = initial_keeper.astype(np.float32)

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

    LOGGER.info("Loading and building datasets.")
    datasets = load_datasets(csv_filepath)
    dataset = build_dataset(datasets)

    LOGGER.info("Removing duplicate text entries.")
    dataset = remove_duplicates(dataset)

    LOGGER.info("Filtering english texts.")
    dataset = filter_english_text(dataset)

    LOGGER.info("Balancing the dataset.")
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
