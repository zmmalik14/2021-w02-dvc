import yaml
import pandas as pd
import argparse


def read_params(config_path):
    """
    This function reads the specified YAML file which contains project configuration.

    :param config_path: the path of the config file to use
    :return: a dictionary containing project configuration options
    """
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)

    return config


def get_data(config_path):
    """
    This function fetches external data from remote storage.

    :param config_path: the path of the config file to use
    :return: a pandas dataframe contains the data
    """
    # Read config file to get external data path
    config = read_params(config_path)
    data_path = config["data_source"]["s3_source"]

    # Read .csv data into pandas dataframe and return it
    df = pd.read_csv(data_path, sep=",", encoding='utf-8')
    return df


if __name__ == "__main__":
    # If the file is being run, parse command line arguments to get config filepath
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()

    # Get the data
    data = get_data(config_path=parsed_args.config)
