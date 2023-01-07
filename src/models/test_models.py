import joblib
import pytest
import pandas as pd
import os
import numpy as np
import config.path_config as path_config
from src.models import predict_model_batch, predict_model_single
from datetime import datetime

MODEL_PATH = os.path.join(
    path_config.ROOT_DIR, "models", "LGBMClassifier_fourth_exp_tuned.joblib"
)
SAMPLE_TEST_PATH = os.path.join(
    path_config.ROOT_DIR, "data", "processed", "processed_test_forth_exp_tuned.csv"
)
DEFAULT_PARAMS_PATH = os.path.join(
    path_config.ROOT_DIR, "src", "experiment_config", "fourth_exp_tuned.yaml"
)

PREDICTION_OUTPUT_FILE = os.path.join(
    path_config.ROOT_DIR,
    "data",
    "prediction",
    f'prediction_sample_{datetime.now().strftime("%m_%d_%Y_%H:%M:%S")}.csv',
)

X_sample1 = pd.read_csv(SAMPLE_TEST_PATH)


@pytest.fixture(scope="module")
def model_load():
    model = joblib.load(MODEL_PATH)
    return model


def ramdomize_col_drop(data: pd.DataFrame):
    cols = list(data.columns)
    random_idx = np.random.randint(low=0, high=len(cols) - 1)
    col_name = cols[random_idx]
    print(col_name)
    return data.drop(col_name, axis=1)


@pytest.mark.parametrize(
    "data,features_in_model,array_out",
    [
        (X_sample1, True, True),
        (ramdomize_col_drop(X_sample1), False, False),
    ],
)
def test_model_output_types(data, features_in_model, array_out, model_load):
    models_feature_names = set(model_load.feature_name_)
    data_features_names = set(list(data.columns))
    compare = models_feature_names.symmetric_difference(data_features_names) == set()
    array_output = (
        type(model_load._validate_data(data, validate_separately=False, reset=False)) == np.array
    )
    assert compare == features_in_model
    assert array_output == array_out


@pytest.mark.parametrize(
    "file_path,output_path,model_path,exists",
    [(SAMPLE_TEST_PATH, PREDICTION_OUTPUT_FILE, MODEL_PATH, True)],
)
def test_model_batch_inference(file_path, output_path, model_path, exists):
    predict_model_batch.batch_inference(
        file_path=file_path, output_path=output_path, model_path=model_path
    )
    assert os.path.exists(output_path) == exists


@pytest.mark.parametrize(
    """ utilization_rate,age,
        number30_59daysdue,
        debtratio,
        monthlyincome,
        numopencredit_loans,
        number90dayslate,
        numberrealestate_loans,
        number60_89daysdue,
        numof_dependents,
        model_path,
        params_path,
        expected_proba,expected_label""",
    [
        (
            0.5,
            20,
            0,
            0.35,
            10_000,
            10,
            2,
            3,
            5,
            3,
            MODEL_PATH,
            DEFAULT_PARAMS_PATH,
            True,
            True,
        ),
    ],
)
def test_model_single_inference(
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
    params_path,
    expected_proba,
    expected_label,
):
    pred_output = predict_model_single.single_inference(
        utilization_rate=utilization_rate,
        age=age,
        number30_59daysdue=number30_59daysdue,
        debtratio=debtratio,
        monthlyincome=monthlyincome,
        numopencredit_loans=numopencredit_loans,
        number90dayslate=number90dayslate,
        numberrealestate_loans=numberrealestate_loans,
        number60_89daysdue=number60_89daysdue,
        numof_dependents=numof_dependents,
        model_path=model_path,
        params_path=params_path,
    )

    exists_output_dict = pred_output["output"]["proba"] is not None
    exists_label_dict = pred_output["output"]["label"] is not None
    assert exists_output_dict == expected_proba
    assert exists_label_dict == expected_label
