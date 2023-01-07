import logging
import sys
import os
import pandas as pd
import argparse
import src.features.feature_eng as feature_eng
import src.data.wrangling as wrangling
import src.utils as utils

logger = logging.getLogger(__name__)
formatter = logging.Formatter(
    fmt="%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

screen_handler = logging.StreamHandler(stream=sys.stdout)
screen_handler.setFormatter(formatter)
logger.addHandler(screen_handler)
logger.setLevel(logging.INFO)

BASE_CONFIG_YML_PATH = "src/experiment_config"


def initialize_argparse():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment_name",
        type=str,
        help="Whether to save model into pickle or not ",
        required=True,
    )
    parser.add_argument(
        "--raw_input_path",
        type=str,
        help="Where the Raw Data Exists ",
        required=True,
    )
    parser.add_argument(
        "--interim_output_path",
        type=str,
        help="Where the interim output file to be stored",
    )
    parser.add_argument(
        "--processed_output_path",
        type=str,
        help="Where the processed (ready) for modelling data stored",
        required=True,
    )
    parser.add_argument(
        "--training_req",
        choices=["True", "False"],
        help="Whether to save model into pickle or not ",
        required=True,
    )
    args = parser.parse_args()
    return args


def build_data(
    raw_input_path: str,
    processed_output_path: str,
    experiment_name: str,
    training_req: bool,
    interim_output_path: os.PathLike = None,
):
    """
    Function to run data preparation before training model

    Args:
        experiment_name (str): experiment_name. Should be the same as the yaml file used
        training_req (os.PathLike): True if you want to create data for training  has target, False if has no target (inference)
        raw_input_path (os.PathLike): Raw File Location
        interim_output_path (os.PathLike, optional): _description_. Defaults to None.
        processed_output_path (os.PathLike): Filename / Path to store processed data





    """
    try:
        logger.info("Making Dataset")
        # read datasource path
        utils.path_checker.csv_extension_checker(raw_input_path)
        data = pd.read_csv(raw_input_path)
        experiment_config_path = os.path.join(
            BASE_CONFIG_YML_PATH, f"{experiment_name}.yaml"
        )
        wrangled_data = wrangling.wrangling_data(data,training_req=training_req,params_path=experiment_config_path)

        # validation_features.validate_wrangling_output_col(
        #     data=wrangled_data, params_path=experiment_config_path
        # )
        if interim_output_path:
            utils.path_checker.csv_extension_checker(interim_output_path)
            wrangled_data.to_csv(
                interim_output_path,index=False
            )  # needs validation ,still assumed the path ext is csv

        feature_engineered_data = feature_eng.feature_engineering_process(data=wrangled_data,params_path=experiment_config_path)
        
        # validation_features.validate_feature_engineering_output_col(
        #     data=feature_engineered_data, 
        # )
        utils.path_checker.csv_extension_checker(processed_output_path)
        feature_engineered_data.to_csv(processed_output_path,index=False)

        logger.info(
            f"""Success Creating Dataset. 
                    Processed Dataset in {processed_output_path}"""
        )
    except BaseException:
        logger.exception("Error Founded at Making Dataset")


if __name__ == "__main__":
    args = initialize_argparse()
    interim_output_path = (
        args.interim_output_path if hasattr(args, "interim_output_path") else None
    )
    build_data(
        raw_input_path=args.raw_input_path,
        processed_output_path=args.processed_output_path,
        experiment_name=args.experiment_name,
        training_req=args.training_req,
        interim_output_path=interim_output_path,
    )
