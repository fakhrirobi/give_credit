import pandas as pd
import logging
import joblib
import sys
import os
import argparse
import config.path_config as path_config
from src.features.feature_eng import feature_engineering_process

DEFAULT_MODEL_PATH = os.path.join(
    path_config.ROOT_DIR, "models", "LGBMClassifier_fourth_exp_tuned.joblib"
)
DEFAULT_PARAMS_PATH = os.path.join(
    path_config.ROOT_DIR, "src", "experiment_config", "fourth_exp_tuned.yaml"
)


# set logging mechanism to inform the progress of data wrangling
logger = logging.getLogger(__name__)
formatter = logging.Formatter(
    fmt="%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

screen_handler = logging.StreamHandler(stream=sys.stdout)
screen_handler.setFormatter(formatter)
logger.addHandler(screen_handler)

logger.setLevel(logging.INFO)


class ModelLoader:
    def __init__(
        self,
        model_path=DEFAULT_MODEL_PATH,
    ) -> None:
        """
        class to load model

        Args:
            model_path (str, optional): ex. Defaults to "give_me_credit/give_me_credit/models/LGBMClassifier_fourth_exp_tuned.joblib".
        """
        self.model_path = model_path

    def load_model(self):
        model = joblib.load(self.model_path)
        return model


def initialize_argparse():
    """
    Argument Parser


    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--utilization_rate",
        type=float,
        required=True,
        help="""
        Total balance on credit cards and personal lines of credit except real estate and
        no installment debt like car loans divided by the sum of credit limits
        """,
    )
    parser.add_argument(
        "--age",
        type=int,
        required=True,
        help="""
                                  Age of borrower in years
                                  """,
    )
    parser.add_argument(
        "--number30_59daysdue",
        type=int,
        required=True,
        help="""Number of times borrower has been 30-59 days past due but no worse in the last 2 years.""",
    )
    parser.add_argument(
        "--monthlyincome",
        type=float,
        required=True,
        help="""
                                  Monthly income
                                  """,
    )
    parser.add_argument(
        "--numopencredit_loans",
        type=int,
        required=True,
        help="""Number of Open loans (installment like car loan or mortgage) and Lines of credit (e.g. credit cards)
            """,
    )
    parser.add_argument(
        "--number90dayslate",
        type=int,
        required=True,
        help="""
            Number of times borrower has been 90 days or more past due.
            """,
    )
    parser.add_argument(
        "--numberrealestate_loans",
        type=int,
        required=True,
        help="""Number of mortgage and real estate loans including home equity lines of credit""",
    )
    parser.add_argument(
        "--number60_89daysdue",
        type=int,
        required=True,
        help="""
                                  Number of times borrower has been 60-89 days past due
                                  but no worse in the last 2 years.""",
    )
    parser.add_argument(
        "--numof_dependents",
        type=int,
        required=True,
        help="""
                                  Number of dependents in family excluding themselves (spouse, children etc.)

                                  """,
    )
    parser.add_argument(
        "--debtratio",
        type=float,
        required=True,
        help="""
                                  Monthly debt payments, alimony,living costs divided by monthy gross income

                                  """,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=False,
        help="Model path to load.Optional default to give_me_credit/give_me_credit/models/LGBMClassifier_fourth_exp_tuned.joblib",
    )
    parser.add_argument(
        "--params_path",
        type=str,
        required=False,
        help="Model path to load.Optional default to give_me_credit/give_me_credit/models/LGBMClassifier_fourth_exp_tuned.joblib",
    )
    args = parser.parse_args()

    return args


def single_inference(
    utilization_rate,
    age,
    number30_59daysdue,
    debtratio,
    monthlyincome,
    numopencredit_loans,
    number90dayslate,
    numberrealestate_loans,
    number60_89daysdue,
    numof_dependents,
    model_path,
    params_path=DEFAULT_PARAMS_PATH,
):
    input_dictionary = {
        "RevolvingUtilizationOfUnsecuredLines": utilization_rate,
        "age": age,
        "NumberOfTime30-59DaysPastDueNotWorse": number30_59daysdue,
        "DebtRatio": debtratio,
        "MonthlyIncome": monthlyincome,
        "NumberOfOpenCreditLinesAndLoans": numopencredit_loans,
        "NumberOfTimes90DaysLate": number90dayslate,
        "NumberRealEstateLoansOrLines": numberrealestate_loans,
        "NumberOfTime60-89DaysPastDueNotWorse": number60_89daysdue,
        "NumberOfDependents": numof_dependents,
    }
    logger.info(f"Feature Input :{input_dictionary} ")
    # read as dataframe
    data = pd.DataFrame(input_dictionary, index=[0])
    feature_generated_data = data.pipe(
        feature_engineering_process, params_path=params_path
    )

    clf_model = ModelLoader(model_path=model_path).load_model()
    prediction_proba = clf_model.predict_proba(feature_generated_data)
    prediction_label = clf_model.predict(feature_generated_data)
    output = {
        "data": input_dictionary,
        "output": {
            "proba": prediction_proba.tolist(),
            "label": prediction_label.tolist(),
        },
    }
    logger.info("output : ", output)
    return output


if __name__ == "__main__":
    args = initialize_argparse()
    single_inference(
        utilization_rate=args.utilization_rate,
        age=args.age,
        number30_59daysdue=args.number30_59daysdue,
        debtratio=args.debtratio,
        monthlyincome=args.monthlyincome,
        numopencredit_loans=args.numopencredit_loans,
        number90dayslate=args.number90dayslate,
        numberrealestate_loans=args.numberrealestate_loans,
        number60_89daysdue=args.number60_89daysdue,
        numof_dependents=args.numof_dependents,
        model_path=args.model_path,
        params_path=args.params_path,
    )
