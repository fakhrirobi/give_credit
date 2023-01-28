import pandas as pd
import numpy as np
import os
import logging
import sys
import joblib
import argparse

from datetime import datetime
import mlflow
import config.path_config as path_config
from mlflow.tracking import MlflowClient
from mlflow.entities import Metric, Param, RunTag
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from config.path_config import ROOT_DIR
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
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

SCORECARD_BASE_DIR = os.path.join(ROOT_DIR,"models")
WOE_REFERENCE_PATH = os.path.join(ROOT_DIR,"models","woe_reference.joblib")
MODEL_BASE_PATH = os.path.join(path_config.ROOT_DIR, "models")

NFOLD = 5
FOLD = StratifiedKFold(NFOLD)
binned_col = ['bin_age',
 'bin_NumberOfDependents',
 'bin_NumberOfTimes90DaysLate',
 'bin_NumberOfTime30-59DaysPastDueNotWorse',
 'bin_NumberOfTime60-89DaysPastDueNotWorse',
 'bin_RevolvingUtilizationOfUnsecuredLines',
 'bin_DebtRatio',
 'bin_MonthlyIncome',
 'bin_NumberOfOpenCreditLinesAndLoans',
 'bin_NumberRealEstateLoansOrLines']

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
    # parser.add_argument(
    #     "--config_path",
    #     type=str,
    #     help="Where the params.yaml path exists ",
    #     required=True,
    # )
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

def create_model_coef_dict(fitted_model,feature_names) : 
    reference = {}
    for x,y in zip (feature_names,fitted_model.coef_.tolist()[0]) : 
        reference[x] = y
    return reference

def create_score_card(model_coef_dict,woe_reference,factor=48.7) :
    temp = []
    for key in woe_reference : 
        data = pd.DataFrame()

        ref = woe_reference.get(key)
        data['bin'] = list(ref.keys())
        data['features'] = key
        data['woe'] = list(ref.values())
        data['woe']= data['woe'].apply(lambda x : x['woe'])
        data['woe'] = data['woe'].replace(np.inf,0)
        data['woe'] = data['woe'].replace(-np.inf,0)
        feature_coef = model_coef_dict.get(key)
        data['score'] = -feature_coef*data['woe']*factor
        data['score'] =data['score'].astype('int')
        temp.append(data)
    scorecard_df = pd.concat(temp)
    scoring_dict = {}
    for feature in scorecard_df.features.unique() : 
        scoring_dict[feature] = {}
        sliced_df = scorecard_df.loc[scorecard_df.features==feature]
        for binrange,score in zip(sliced_df['bin'],sliced_df['score']) : 
            scoring_dict[feature][binrange] =score
    return scorecard_df,scoring_dict


def train_model(training_path, experiment_name):

    try:
        logger.info("Training Started")

        client = MlflowClient()

        run = mlflow.start_run(run_name=experiment_name)
        
        # mlflow.set_tracking_uri("http://127.0.0.1:5000")
        # read data

        path_checker.csv_extension_checker(training_path)
        mlflow.log_artifact(training_path)
        train_data = pd.read_csv(training_path)

        # split X and Y
        woe_features = [x for x in train_data.columns if x.startswith('woe')]
        X = train_data[woe_features]
        Y = train_data[TARGET]
        # replace params from yaml_file
        # path_checker.yaml_extension_checker(config_path)
        # params = (
        #     load_yaml(config_path)["params"]
        #     if "params" in load_yaml(config_path).keys()
        #     else {}
        # )
        # if
        clf_model = LogisticRegression(class_weight="balanced")
        cross_val_result_text, cross_val_metric = begin_cross_validation(
            X, Y, clf_model, FOLD
        )
        logger.info("CV Result, AUC {}".format(cross_val_result_text))
        model_params = clf_model.get_params()
        # train model with all data
        clf_model.fit(X, Y)
        model_coef = create_model_coef_dict(fitted_model=clf_model,feature_names=binned_col)
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
        

        woe_reference_loaded = joblib.load(WOE_REFERENCE_PATH)
        #store model coef 
        scorecard_df,scoring_dict =create_score_card(model_coef_dict=model_coef,woe_reference=woe_reference_loaded)
        scorecard_df.to_excel(os.path.join(SCORECARD_BASE_DIR,"scorecard.xlsx"),index=False)
        joblib.dump(scoring_dict,os.path.join(SCORECARD_BASE_DIR,"scorecard_dict.joblib"))
        
        #log the artifact 
        mlflow.log_artifact(MODEL_SAVE_PATH)
        mlflow.log_artifact(WOE_REFERENCE_PATH)
        mlflow.log_artifact(os.path.join(SCORECARD_BASE_DIR,"scorecard.xlsx"))
        mlflow.log_artifact(os.path.join(SCORECARD_BASE_DIR,"scorecard_dict.joblib"))
        # if config_path:
        #     mlflow.log_artifact(config_path)
        logger.info("SUCESSFULLY SAVED MODEL")

    except BaseException:
        logger.exception("Error Occured During Training Logistic Model")


if __name__ == "__main__":
    args = initialize_argparse()

    train_model(
        training_path=args.training_data_path,
        experiment_name=args.experiment_name,
    )
    