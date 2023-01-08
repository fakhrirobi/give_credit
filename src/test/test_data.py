import pytest
import os
import pandas as pd
import numpy as np
import src.data.wrangling as wrangling
import config.path_config as path_config

INTERIM_TEST_DATA = os.path.join(
    path_config.ROOT_DIR, "data", "interim", "interim_test_forth_exp_tuned.csv"
)
INTERIM_TRAINING_DATA = os.path.join(
    path_config.ROOT_DIR, "data", "interim", "interim_training_forth_exp_tuned.csv"
)

RAW_TRAINING_DATA_PATH = os.path.join(
    path_config.ROOT_DIR, "data", "raw", "cs-training.csv"
)
RAW_TEST_DATA_PATH = os.path.join(path_config.ROOT_DIR, "data", "raw", "cs-test.csv")
data1 = pd.read_csv(RAW_TRAINING_DATA_PATH)
data2 = pd.read_csv(RAW_TEST_DATA_PATH)


PARAMS_PATH = os.path.join(
    path_config.ROOT_DIR, "src", "experiment_config", "fourth_exp_tuned.yaml"
)


def ramdomize_col_drop(data: pd.DataFrame):
    cols = list(data.columns)
    random_idx = np.random.randint(low=0, high=len(cols) - 1)
    col_name = cols[random_idx]
    print(col_name)
    return data.drop(col_name, axis=1)


@pytest.mark.parametrize(
    "data,training_req,expected_output",
    [
        (data1, True, True),
        (data2, False, True),
        (ramdomize_col_drop(data1), True, False),
    ],
)
def test_wrangling_data(data, training_req, expected_output):
    assert (
        isinstance(
            wrangling.wrangling_data(
                data, training_req=training_req, params_path=PARAMS_PATH
            ),
            pd.core.frame.DataFrame,
        )
        == expected_output
    ), f"Output : {wrangling.wrangling_data(data, training_req=training_req, params_path=PARAMS_PATH)}"
