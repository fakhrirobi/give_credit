import pandas as pd
import logging
import joblib
import sys
import argparse


DEFAULT_MODEL_PATH = "/models/LGBMClassifier_fourth_exp_tuned.joblib"
from src.features.feature_eng import feature_engineering_process
import src.utils.load_config as load_config
import src.utils.path_checker as path_checker

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
        model_path="give_me_credit/give_me_credit/models/LGBMClassifier_fourth_exp_tuned.joblib",
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
    subparsers = parser.add_subparsers()
    
    single_inference = subparsers.add_parser("single_inference")
    single_inference.add_argument(
        "--utilization_rate",
        type=float,
        required=True,
        help="""
                                  Total balance on credit cards and personal lines of credit except real estate and 
                                  no installment debt like car loans divided by the sum of credit limits
                                  
                                  """,
    )
    single_inference.add_argument(
        "--age",
        type=int,
        required=True,
        help="""
                                  Age of borrower in years
                                  """,
    )
    single_inference.add_argument(
        "--number30_59daysdue",
        type=int,
        required=True,
        help=""" 
                                  Number of times borrower has been 30-59 days past due but no worse in the last 2 years.

                                  """,
    )
    single_inference.add_argument(
        "--monthlyincome",
        type=float,
        required=True,
        help="""
                                  Monthly income
                                  """,
    )
    single_inference.add_argument(
        "--numopencredit_loans",
        type=int,
        required=True,
        help="""
                                  
                                  Number of Open loans (installment like car loan or mortgage) and 
                                  Lines of credit (e.g. credit cards)

                                  
                                  
                                  """,
    )
    single_inference.add_argument(
        "--number90dayslate",
        type=int,
        required=True,
        help="""
                                  Number of times borrower has been 90 days or more past due.

                                  
                                  """,
    )
    single_inference.add_argument(
        "--numrealestate_loans",
        type=int,
        required=True,
        help="""
                                  Number of mortgage and real estate loans 
                                  including home equity lines of credit

                                  
                                  """,
    )
    single_inference.add_argument(
        "--number60_89daysdue",
        type=int,
        required=True,
        help="""
                                  Number of times borrower has been 60-89 days past due 
                                  but no worse in the last 2 years.

                                  """,
    )
    single_inference.add_argument(
        "--numof_dependents",
        type=int,
        required=True,
        help="""
                                  Number of dependents in family excluding themselves (spouse, children etc.)

                                  """,
    )
    single_inference.add_argument(
        "--debtratio",
        type=float,
        required=True,
        help="""
                                  Monthly debt payments, alimony,living costs divided by monthy gross income

                                  """,
    )
    single_inference.add_argument(
        '--model_path',
        type=str,
        required=False,
        help='Model path to load.Optional default to give_me_credit/give_me_credit/models/LGBMClassifier_fourth_exp_tuned.joblib'
    )
    batch_inference = subparsers.add_parser("batch_inference")
    batch_inference.add_argument("--file_path", type=str, required=True)
    batch_inference.add_argument("--output_path", type=str, required=True)
    batch_inference.add_argument(
        '--model_path',
        type=str,
        required=False,
        help='Model path to load.Optional default to give_me_credit/give_me_credit/models/LGBMClassifier_fourth_exp_tuned.joblib'
    )
    args = parser.parse_args()

    return args


def inference(args):
    if hasattr(args, "utilization_rate") == True:
        input_dictionary = {
            "RevolvingUtilizationOfUnsecuredLines": args.utilization_rate,
            "age": args.age,
            "NumberOfTime30-59DaysPastDueNotWorse": args.number30_59daysdue,
            "DebtRatio": args.debtratio,
            "MonthlyIncome": args.monthlyincome,
            "NumberOfOpenCreditLinesAndLoans": args.numopencredit_loans,
            "NumberOfTimes90DaysLate": args.number90dayslate,
            "NumberRealEstateLoansOrLines": args.numrealestate_loans,
            "NumberOfTime60-89DaysPastDueNotWorse": args.number60_89daysdue,
            "NumberOfDependents": args.numof_dependents,
        }
        logger.info(f"Feature Input :{input_dictionary} ")
        # read as dataframe
        data = pd.read_dict(input_dictionary)
        feature_generated_data = data.pipe(feature_engineering_process)

        clf_model = ModelLoader(model_path="/").load_model()
        prediction_proba = clf_model.predict_proba(feature_generated_data)
        return prediction_proba  # should be json format

    if hasattr(args, "file_path") == True:
        path_checker.check_is_file(args.file_path, create_file=False)
        path_checker.csv_extension_checker(args.file_path)

        feature = pd.read_csv(args.file_path)
        model_path = args.model_path if hasattr(args,"model_path") else DEFAULT_MODEL_PATH
        clf_model = ModelLoader(model_path=model_path).load_model()
        prediction_proba = clf_model.predict_proba(feature.to_numpy())[:, 1]
        prediction_df = feature
        prediction_df.loc[:, "Probability"] = prediction_proba

        path_checker.check_is_file(path=args.output_path, create_file=True)
        path_checker.csv_extension_checker(path=args.output_path)

        prediction_df.to_csv(args.output_path, index=False)
        logger.info(f"Sucessfully Saved Prediction file in {args.output_path}")


if __name__ == "__main__":
    args = initialize_argparse()
    inference(args=args)
