import pandas as pd
import numpy as np
import os
import logging
import joblib
import sys
import mlflow 
from  mlflow.tracking import MlflowClient
from mlflow.entities import Metric, Param, RunTag
import yaml
import argparse
from datetime import datetime
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn import metrics
from config.path_config import ROOT_DIR
import optuna
from optuna.visualization import plot_intermediate_values
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
    parser.add_argument(
        "--experiment_name",
        type=str,
        help="Experiment name to log ",
        required=True,
    )

    parser.add_argument(
        "--training_data_path",
        type=str,
        help="Where the training path exists ",
        required=True,
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        help="Input number of trials (optuna parameter) ",
        required=True,
    )
    args = parser.parse_args()
    return args




MODEL_BASE_PATH = os.path.join(ROOT_DIR,"src","models","tuning_result")


PROCESSED_BASE_PATH = os.path.join(ROOT_DIR,"data","processed")
TRAINING_PROCESSED_PATH = os.path.join(PROCESSED_BASE_PATH, "training_processed.csv")
TEST_PROCESSED_PATH = os.path.join(PROCESSED_BASE_PATH, "test_processed.csv")


TARGET = "SeriousDlqin2yrs"
NFOLD = 5

def mlflow_experiment(experiment_name) : 
    client = MlflowClient()
    experiment_search_result = client.search_experiments(filter_string=f"attribute.name = '{experiment_name}'").to_list()[0]
    print(experiment_search_result)
    if experiment_search_result == list() : 
        experiment = client.create_experiment(name=experiment_name)
        return client,experiment.to_list()[0]
    return  client,experiment_search_result

def run_study(experiment_name,training_data_path,num_trials) : 
    client,experiment= mlflow_experiment(experiment_name)
    run = client.create_run(experiment_id=experiment.experiment_id)
    func = lambda trial: objective(trial,training_data_path)
    study = optuna.create_study(direction="maximize")
    study.optimize(func, n_trials=num_trials)
    best_params = study.best_params
    best_auc = study.best_value
    return best_params,best_auc,client,run

def objective(trial,training_data_path):
    try:

        param = {
            "objective": "binary",
            "class_weight":"balanced",
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            "max_depth": trial.suggest_int("max_depth", 2, 30),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.2, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 300),
        }

        train_data = pd.read_csv(training_data_path)
        # split X and Y
        X = train_data.drop(TARGET, axis=1)
        Y = train_data[TARGET]

        scoring_dict = {"auc": []}

        fold = StratifiedKFold(NFOLD)
        for train_idx, test_idx in fold.split(X, Y):
            train_x, val_x = X.iloc[train_idx], X.iloc[test_idx]
            train_y, val_y = Y.iloc[train_idx], Y.iloc[test_idx]
            # train model

            clf_model = LGBMClassifier(**param)
            model = CalibratedClassifierCV(clf_model, cv=fold, method="isotonic",ensemble=True)
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
    except BaseException:
        logger.exception("PROCESS TERMINATED : at {__name__}")


if __name__ == "__main__":
    try:
        args = initialize_argparse()

    
        
        
        #yaml file for best params
        timestamp = datetime.now().strftime("%m%d%Y%H%M%S")
        PARAMS_NAME = f"best_params_lgbm_{timestamp}.yaml"
        yaml_best_params_file_path = os.path.join(MODEL_BASE_PATH, PARAMS_NAME)
        
        best_params,best_auc,client,run = run_study(experiment_name=args.experiment_name,
                  training_data_path=args.training_data_path,num_trials=args.num_trials
                )
        tags = {
        "experiment_name": args.experiment_name,
        "timestamp": datetime.now().strftime("%m_%d_%Y_%H:%M:%S")}
        val_metrics = [Metric("auc", best_auc, step=1, timestamp=0)]
        logged_params = [Param(p, f"{best_params[p]}") for p in best_params.keys()]
        logged_tags = [RunTag(t, f"{tags[t]}") for t in tags.keys()]
        client.log_batch(
            run_id=run.info.run_id,
            metrics=val_metrics,
            params=logged_params,
            tags=logged_tags,
        )

        with open(yaml_best_params_file_path,'w') as output_path : 
            yaml.dump(best_params,output_path)
        client.log_artifact(run.info.run_id,yaml_best_params_file_path)

        
    except BaseException:

        logger.exception("PROCESS TERMINATED : at {__name__}")
        