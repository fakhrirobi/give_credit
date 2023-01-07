import pandas as pd
import numpy as np
import os
import logging
import sys
import joblib
import yaml
import argparse

from datetime import datetime
import mlflow
import config.path_config as path_config
from mlflow.tracking import MlflowClient
from mlflow.entities import Metric, Param, RunTag
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

from lightgbm import LGBMClassifier

from src.utils.load_config import load_yaml
from src.utils import path_checker


logger = logging.getLogger(__name__)
formatter = logging.Formatter(
    fmt="%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

screen_handler = logging.StreamHandler(stream=sys.stdout)
screen_handler.setFormatter(formatter)

logger.addHandler(screen_handler)
logger.setLevel(logging.INFO)


TARGET = "SeriousDlqin2yrs"



MODEL_BASE_PATH = os.path.join(path_config.ROOT_DIR,'models')

NFOLD = 5
FOLD = StratifiedKFold(NFOLD)


def initialize_argparse():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment_name",
        type=str,
        help="Whether to save model into pickle or not ",
        required=True,
    )
    parser.add_argument(
        "--training_data_path",
        type=str,
        help="Where the training path exists ",
        required=True,
    )
    parser.add_argument(
        "--config_path",
        type=str,
        help="Where the params.yaml path exists ",
        required=True,
    )
    args = parser.parse_args()
    return args


def begin_cross_validation(X, Y, model, fold):
    scoring_dict = {"auc": []}
    for train_idx, test_idx in fold.split(X, Y):
        train_x, val_x = X.iloc[train_idx], X.iloc[test_idx]
        train_y, val_y = Y.iloc[train_idx], Y.iloc[test_idx]
        # train model

        model.fit(train_x, train_y)
        y_pred = model.predict_proba(val_x)
        # calculate metrics
        scoring_dict["auc"].append(metrics.roc_auc_score(val_y, y_pred[:, 1]))

    # print the scoring
    # generate some picture of ROC AUC Curve
    # log the picture to artifact
    result = "Scoring: AUC :{}".format(np.mean(scoring_dict["auc"]))
    return result, np.mean(scoring_dict["auc"])


def search_run_or_create_run(experiment_name):
    if mlflow.search_runs(filter_string=f"={experiment_name}"):
        run = mlflow.search_runs(
            filter_string=f"""attributes.run_name = '{experiment_name}'"""
        )
        return run
    else:
        run = mlflow.start_run(run_name=experiment_name)
        return run


def train_model(training_path, experiment_name, config_path):

    try:
        logger.info("Training Started")

        client = MlflowClient()

        run = mlflow.start_run(run_name=experiment_name)
        run = search_run_or_create_run(experiment_name=experiment_name)
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        # read data

        path_checker.check_is_file(training_path, create_file=False)
        path_checker.csv_extension_checker(training_path)
        mlflow.log_artifact(training_path)
        train_data = pd.read_csv(training_path)

        # split X and Y
        X = train_data.drop(TARGET, axis=1)
        Y = train_data[TARGET]
        # replace params from yaml_file
        path_checker.check_is_file(config_path)
        path_checker.yaml_extension_checker(config_path)
        params = (
            load_yaml(config_path)["params"]
            if "params" in load_yaml(config_path).keys()
            else {}
        )
        # if
        clf_model = LGBMClassifier(**params)
        cross_val_result_text, cross_val_metric = begin_cross_validation(
            X, Y, clf_model, FOLD
        )
        logger.info("CV Result, AUC {}".format(cross_val_result_text))
        model_params = clf_model.get_params()
        # train model with all data
        clf_model.fit(X, Y)
        model_name = type(clf_model).__name__

        tags = {
            "experiment_name": experiment_name,
            "timestamp": datetime.now().strftime("%m_%d_%Y_%H:%M:%S"),
        }
        val_metrics = [Metric("auc", cross_val_metric, step=1, timestamp=0)]
        logged_params = [Param(p, f"{model_params[p]}") for p in model_params.keys()]
        logged_tags = [RunTag(t, f"{tags[t]}") for t in tags.keys()]
        client.log_batch(
            run_id=run.info.run_id,
            metrics=val_metrics,
            params=logged_params,
            tags=logged_tags,
        )

        MODEL_FORMAT_NAME = "{}_{}.joblib".format(model_name, experiment_name)
        MODEL_SAVE_PATH = os.path.join(MODEL_BASE_PATH, MODEL_FORMAT_NAME)
        joblib.dump(clf_model, MODEL_SAVE_PATH)

        mlflow.log_artifact(MODEL_SAVE_PATH)

        if config_path:
            mlflow.log_artifact(config_path)
        logger.info("SUCESSFULLY SAVED MODEL")

    except BaseException as error:
        logger.exception("Error Occured During Training")


if __name__ == "__main__":
    args = initialize_argparse()

    train_model(
        training_path=args.training_data_path,
        experiment_name=args.experiment_name,
        config_path=args.config_path,
    )
