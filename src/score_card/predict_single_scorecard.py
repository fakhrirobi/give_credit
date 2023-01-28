import pandas as pd
import logging
import joblib
import sys
import os
import argparse
import config.path_config as path_config
from src.score_card.make_feature import feature_engineering
from src.score_card.turn_scorecard import calculate_scorecard_from_df
DEFAULT_MODEL_PATH = os.path.join(
    path_config.ROOT_DIR, "models", "LogisticRegression_scorecard_exp.joblib"
)
DEFAULT_PARAMS_PATH = os.path.join(
    path_config.ROOT_DIR, "src", "experiment_config", "fourth_exp_tuned.yaml"
)


# set logging mechanism to inform the progress of data wrangling
logger = logging.getLogger(__name__)
formatter = logging.Formatter(
    fmt="%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

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

    args = parser.parse_args()

    return args


def single_inference_scorecard(
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
    model_path
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
    
    #implementation of binning 
    feature_generated_data = feature_engineering(data=data,training_req=False)
    print("feature_generated_data",feature_generated_data.columns)
    #calculate credit score 
    scored_data = calculate_scorecard_from_df(data=feature_generated_data,binned_col=binned_col)
    woe_features = [x for x in scored_data.columns if x.startswith('woe')]
    print('feature_generated_data[woe_features]',woe_features)
    clf_model = ModelLoader(model_path=model_path).load_model()
    prediction_proba = clf_model.predict_proba(feature_generated_data[woe_features])
    prediction_label = clf_model.predict(feature_generated_data[woe_features])
    credit_score_int = scored_data['score'].squeeze().tolist()
    print('credit score',type(credit_score_int))
    decision_credit = 'Approved' if scored_data['score'].squeeze() >=531 else 'Rejected' 
    print('credit_decision',decision_credit)
    output = {
        "data": input_dictionary,
        "output": {
            "proba": prediction_proba.tolist(),
            "label": prediction_label.tolist(),
            "credit_score" : credit_score_int,
            "accept_credit" : decision_credit
        },
    }
    logger.info("output : ", output)
    return output


if __name__ == "__main__":
    args = initialize_argparse()
    single_inference_scorecard(
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
        model_path=args.model_path
    )
