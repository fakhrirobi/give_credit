import pandas as pd
import logging
import joblib
import sys
import os
import argparse
import config.path_config as path_config
import src.utils.path_checker as path_checker

DEFAULT_MODEL_PATH = os.path.join(
    path_config.ROOT_DIR, "models", "LGBMClassifier_fourth_exp_tuned.joblib"
)
DEFAULT_PARAMS_PATH = os.path.join(
    path_config.ROOT_DIR, "src", "experiment_config", "fourth_exp_tuned.yaml"
)


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

    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument(
        "--model_path",
        type=str,
        required=False,
        help="Model path to load.Optional default to give_me_credit/give_me_credit/models/LGBMClassifier_fourth_exp_tuned.joblib",
    )
    args = parser.parse_args()

    return args


def batch_inference(file_path, output_path, model_path=DEFAULT_MODEL_PATH):
    path_checker.csv_extension_checker(file_path)

    feature = pd.read_csv(file_path)
    clf_model = ModelLoader(model_path=model_path).load_model()
    prediction_proba = clf_model.predict_proba(feature.to_numpy())[:, 1]
    prediction_df = feature
    prediction_df.loc[:, "Probability"] = prediction_proba

    path_checker.csv_extension_checker(path=output_path)

    prediction_df.to_csv(output_path, index=False)
    logger.info(f"Sucessfully Saved Prediction file in {output_path}")


if __name__ == "__main__":
    args = initialize_argparse()
    batch_inference(
        input_path=args.input_path,
        output_path=args.output_path,
        model_path=args.model_path,
    )
