import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from get_data import read_params


def split_and_saved_data(config_path):
    """
    This function splits raw data into training and validation sets.

    :param config_path: the path of the config file to use
    """
    # Read configuration options
    config = read_params(config_path)
    test_data_path = config["split_data"]["test_path"]
    train_data_path = config["split_data"]["train_path"]
    raw_data_path = config["load_data"]["raw_dataset_csv"]
    split_ratio = config["split_data"]["test_size"]
    random_state = config["base"]["random_state"]

    # Read in the raw dataset and split it into train and test sets
    df = pd.read_csv(raw_data_path, sep=",")
    train, test = train_test_split(
        df,
        test_size=split_ratio,
        random_state=random_state
    )

    # Export train and test datasets to individual .csv files
    train.to_csv(train_data_path, sep=",", index=False, encoding="utf-8")
    test.to_csv(test_data_path, sep=",", index=False, encoding="utf-8")


if __name__ == "__main__":
    # If the file is being run, parse command line arguments to get config filepath
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()

    # Split raw data into training and validation (test) sets
    split_and_saved_data(config_path=parsed_args.config)
