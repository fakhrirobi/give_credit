import pandas as pd
import numpy as np
import os
import logging
import joblib
import sys
import argparse
from datetime import datetime
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

import optuna

# set logging mechanism to inform the progress of data wrangling
logger = logging.getLogger(__name__)
formatter = logging.Formatter(
    fmt="%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

screen_handler = logging.StreamHandler(stream=sys.stdout)
screen_handler.setFormatter(formatter)
logger.addHandler(screen_handler)

logger.setLevel(logging.DEBUG)


def initialize_argparse():
    parser = argparse.ArgumentParser()
    # adding argument for each method
    parser.add_argument(
        "--num_trials",
        type=int,
        help="Input number of trials (optuna parameter) ",
        required=True,
    )
    args = parser.parse_args()
    return args


MODEL_BASE_PATH = "give_me_credit/give_me_credit/models/"


PROCESSED_BASE_PATH = "../give_me_credit/data/processed/"
TRAINING_PROCESSED_PATH = os.path.join(PROCESSED_BASE_PATH, "training_processed.csv")
TEST_PROCESSED_PATH = os.path.join(PROCESSED_BASE_PATH, "test_processed.csv")


TARGET = "SeriousDlqin2yrs"
NFOLD = 5


def objective(trial):
    try:
        param = {
            "objective": "binary",
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            "max_depth": trial.suggest_int("max_depth", 2, 30),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.2, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 200),
        }

        train_data = pd.read_csv(TRAINING_PROCESSED_PATH)
        # split X and Y
        X = train_data.drop(TARGET, axis=1)
        Y = train_data[TARGET]

        scoring_dict = {"auc": []}

        fold = StratifiedKFold(NFOLD)
        for train_idx, test_idx in fold.split(X, Y):
            train_x, val_x = X.iloc[train_idx], X.iloc[test_idx]
            train_y, val_y = Y.iloc[train_idx], Y.iloc[test_idx]
            # train model

            model = LGBMClassifier(**param)
            model.fit(train_x, train_y)

            y_pred = model.predict_proba(val_x)
            # calculate metrics
            scoring_dict["auc"].append(metrics.roc_auc_score(val_y, y_pred[:, 1]))

        # print the scoring
        print(
            f"""Scoring: AUC :{np.mean(scoring_dict['auc'])}
        """
        )

        return np.mean(scoring_dict["auc"])
    except BaseException as error:
        logger.error("PROCESS TERMINATED : {}".format(str(error)))


if __name__ == "__main__":
    try:
        method_args = initialize_argparse()

        timestamp = datetime.now().strftime("%m%d%Y%H%M%S")
        argument_parser = initialize_argparse()
        PARAMS_NAME = f"best_params_lgbm_{timestamp}.joblib"
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=100)
        best_params = study.best_params
        joblib.dump(best_params, os.path.join(MODEL_BASE_PATH, PARAMS_NAME))
        # logger.info(f'Best Params')
        # logger.info(f'Tuning Process Success Params is in {best_params}')
    except BaseException as error:
        print(str(error))
        # logger.error("PROCESS TERMINATED : {}".format(str(error)))
