from get_data import read_params, get_data
import argparse


def load_and_save(config_path):
    """
    This function loads and saves raw data from remote storage.

    :param config_path: the path of the config file to use
    """
    # Read configuration options from config file
    config = read_params(config_path)

    # Get data from remote storage and drop rows with missing values
    df = get_data(config_path)
    df = df.dropna()

    # Replace spaces in column names with underscores
    new_cols = [col.replace(" ", "_") for col in df.columns]

    # Get raw dataset filepath from config options and save the data
    raw_data_path = config["load_data"]["raw_dataset_csv"]
    df.to_csv(raw_data_path, sep=",", index=False, header=new_cols)


if __name__ == "__main__":
    # If the file is being run, parse command line arguments to get config filepath
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()

    # Load data from remote storage and save to the configured raw dataset filepath
    load_and_save(config_path=parsed_args.config)
