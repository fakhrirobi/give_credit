import pandas as pd
import numpy as np
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


# list of transformation
def log_debt_ratio(data, col="DebtRatio"):
    return np.log1p(data["DebtRatio"])


def log_income(data, col="MonthlyIncome"):
    return np.log1p(data["MonthlyIncome"])


def log_revolvingrate(data, col="RevolvingUtilizationOfUnsecuredLines"):
    return np.log1p(data["RevolvingUtilizationOfUnsecuredLines"])


def feature_engineering_process(data,params_path):
    logger.info("PROCESS STARTED")
    try:
        data["LogDebtRatio"] = data.pipe(log_debt_ratio, col="DebtRatio")

        data["LogIncome"] = data.pipe(log_income, col="MonthlyIncome")

        data["LogRevolvingUtilizationOfUnsecuredLines"] = data.pipe(
            log_revolvingrate, col="RevolvingUtilizationOfUnsecuredLines"
        )

        data.drop(
            [
                "DebtRatio",
                "RevolvingUtilizationOfUnsecuredLines",
                "MonthlyIncome",
            ],
            axis=1,
            inplace=True,
        )
        validation_features.validate_feature_engineering_output_col(data=data,params_path=params_path)
        return data
       
    except BaseException as error:
        logger.exception("Error Encountered at Feature Engineering Steps")

    logger.info("PROCESS ENDED SUCESSFULLY")
