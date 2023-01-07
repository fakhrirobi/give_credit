import pandas as pd
import os
import logging
import sys

import src.features.validation_features as validation_features

# set logging mechanism to inform the progress of data wrangling
logger = logging.getLogger(__name__)
formatter = logging.Formatter(
    fmt="%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

screen_handler = logging.StreamHandler(stream=sys.stdout)
screen_handler.setFormatter(formatter)
logger.addHandler(screen_handler)

logger.setLevel(logging.INFO)


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


def wrangling_data(data:pd.DataFrame,params_path:str,training_req:bool=False):
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
        
        data = data.drop_duplicates()
        
        data["MonthlyIncome"] = data["MonthlyIncome"].fillna(
            data["MonthlyIncome"].median()
        )
        data["NumberOfDependents"] = data["NumberOfDependents"].fillna(
            data["NumberOfDependents"].mode()[0]
        )
        
        data = data[TRAIN_COLUMNS] if training_req==True else  data[TEST_COLUMNS]
        print(data.isna().any())
        for col in data.columns : 
            if data[col].isna().any() == True : 
                raise ValueError(f'During Data Wrangling There is Missing Value on Columns : {col}.Fix the pipeline by adding imputer / imputation strategy ') 
        

        
        
        # assert that train data contain certain columns
        if training_req==True : 
            if "SeriousDlqin2yrs" not in data.columns.tolist() :
                raise AssertionError("Train Columns Doesnot contains TARGET Column : SeriousDlqin2yrs")
            return data
        validation_features.validate_wrangling_output_col(data=data,params_path=params_path)
        return data
        
    except BaseException as error:
        logger.exception(f"Process Encountered Errot at Data Wrangling Process : {error}")

    logger.info("PROCESS ENDED SUCESSFULLY")

