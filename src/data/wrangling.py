import pandas as pd
import logging
import sys
import os
import joblib 
import src.features.validation_features as validation_features
from config.path_config import ROOT_DIR
# set logging mechanism to inform the progress of data wrangling
logger = logging.getLogger(__name__)
formatter = logging.Formatter(
    fmt="%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

screen_handler = logging.StreamHandler(stream=sys.stdout)
screen_handler.setFormatter(formatter)
logger.addHandler(screen_handler)

logger.setLevel(logging.INFO)
IMPUTER_BASE_DIR = os.path.join(ROOT_DIR,"src","features","imputer")

TRAIN_COLUMNS = [
    "SeriousDlqin2yrs",
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]
TEST_COLUMNS = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

median_income_imputer_path = os.path.join(IMPUTER_BASE_DIR,"median_income_imputer.joblib")
mode_num_dep_path = os.path.join(IMPUTER_BASE_DIR,"number_of_dependents_mode_imputer.joblib")

def wrangling_data(data: pd.DataFrame, params_path: str, training_req: bool):
    """_summary_

    Args:
        data (pd.DataFrame): _description_
        training_req (bool, optional): _description_. Defaults to False.

    Raises:
        AssertionError: _description_
        AssertionError: _description_
    """
    logger.info("PROCESS STARTED")
    try:
        # the problem we face with the data wrangling is only with the missing values
        print('wrangling position cols beginning',data.columns)
        data = data.drop_duplicates()
        if training_req == "True" : 
            median_income = data["MonthlyIncome"].median()
            
            number_of_dependents_mode = data["NumberOfDependents"].mode()[0]
            data["MonthlyIncome"] = data["MonthlyIncome"].fillna(
                median_income
            )
            data["NumberOfDependents"] = data["NumberOfDependents"].fillna(
                number_of_dependents_mode
            )
            joblib.dump(median_income,os.path.join(IMPUTER_BASE_DIR,"median_income_imputer.joblib"))
            joblib.dump(number_of_dependents_mode,os.path.join(IMPUTER_BASE_DIR,"number_of_dependents_mode_imputer.joblib"))
            
        if training_req == "False" : 
            if os.path.exists(median_income_imputer_path) == False : 
                raise ValueError("Median Income Imputer Has Not Been Dumped.Create Training Data First")
            if os.path.exists(mode_num_dep_path) == False : 
                raise ValueError("Number of Dependents Mode Imputer Has Not Been Dumped.Create Training Data First")
            
            median_income = joblib.load(median_income_imputer_path)
            number_of_dependents_mode = joblib.load(mode_num_dep_path)      
            data["MonthlyIncome"] = data["MonthlyIncome"].fillna(
                median_income
            )
            data["NumberOfDependents"] = data["NumberOfDependents"].fillna(
                number_of_dependents_mode
            )
            
            
        col_arrangement = {
            "True" : TRAIN_COLUMNS,
            "False" : TEST_COLUMNS
        }
        
        features_to_take = col_arrangement.get(training_req)
        data = data.loc[:,features_to_take]
        print('wrangling position cols',data.columns)
        for col in data.columns:
            if data[col].isna().any() == True:
                raise ValueError(
                    f"During Data Wrangling There is Missing Value on Columns : {col}.Fix the pipeline by adding imputer / imputation strategy "
                )

        # assert that train data contain certain columns
        if training_req == "True":
            if "SeriousDlqin2yrs" not in data.columns.tolist():
                raise AssertionError(
                    "Train Columns Doesnot contains TARGET Column : SeriousDlqin2yrs"
                )
            return data
        validation_features.validate_wrangling_output_col(
            data=data, params_path=params_path
        )
        return data

    except BaseException as error:
        logger.exception(
            f"Process Encountered Errot at Data Wrangling Process : {error}"
        )

