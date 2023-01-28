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

SCORECARD_BASE_DIR = os.path.join(ROOT_DIR,"src","models")
SCORECARD_DICTIONARY_VALUES = os.path.join(SCORECARD_BASE_DIR,"scorecard_dict.joblib")
MODEL_BASE_PATH = os.path.join(path_config.ROOT_DIR, "models")
CREDIT_SCORE_BASEPATH = os.path.join(path_config.ROOT_DIR, "data","credit_score")

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
        "--filename",
        type=str,
        help="Filename of Scored Data ",
        required=True,
    )
    parser.add_argument(
        "--score_data",
        type=str,
        help="Where the feature contain score_data",
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


def apply_scoring(data,scoring_dict,feature_names:list,offset=650) : 
    for feature in feature_names : 
        #create new feature scorer 
        feature_dict = scoring_dict.get(feature)
        data[f'score_{feature}'] =data[feature].apply(lambda x:feature_dict.get(x))
        score_col = [x for x in data.columns if x.startswith('score_')]
        data['score'] = data[score_col].sum(axis=1) + offset
        data.drop(score_col,axis=1,inplace=True)
    return data

def calculate_scorecard_from_csv(score_data_path,filename):

    try:
        logger.info("Scoring Started")
        score_data = pd.read_csv(score_data_path)
        
        #feature validation with bin columns 
        
        #load scoring dictionary 
        scoring_dict_loaded = joblib.load(SCORECARD_DICTIONARY_VALUES)
        scored_data = apply_scoring(data=score_data,scoring_dict=scoring_dict_loaded,feature_names=binned_col)
        
        if filename : 
            score_data.to_csv(f'{os.path.join(CREDIT_SCORE_BASEPATH,filename)}.csv',index=False)
        return score_data
        # if config_path:
        #     mlflow.log_artifact(config_path)
        logger.info(f"SUCESSFULLY CALCULATED SCORE .Scored data saved in {os.path.join(CREDIT_SCORE_BASEPATH,filename)}.csv")

    except BaseException:
        logger.exception("Error Occured During Scoring ")

def calculate_scorecard_from_df(data:pd.DataFrame,binned_col):

    try:
        logger.info("Scoring Started")
        score_data = data
        
        #feature validation with bin columns 
        
        #load scoring dictionary 
        scoring_dict_loaded = joblib.load(SCORECARD_DICTIONARY_VALUES)
        scored_data = apply_scoring(data=score_data,scoring_dict=scoring_dict_loaded,feature_names=binned_col)
        
        logger.info(f"SUCESSFULLY CALCULATED CREDIT SCORE")
        return scored_data
        # if config_path:
        #     mlflow.log_artifact(config_path)


    except BaseException:
        logger.exception("Error Occured During Scoring ")


if __name__ == "__main__":
    args = initialize_argparse()

    calculate_scorecard_from_csv(
        score_data_path=args.score_data,
        filename=args.filename
    )
    