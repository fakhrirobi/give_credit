import pytest
import pandas as pd 
import numpy as np
import src.data.wrangling as wrangling
import src.features.feature_eng as feature_eng
import src.features.validation_features as validation_features

RAW_TRAINING_DATA_PATH = 'give_credit/data/raw/cs-training.csv'
RAW_TEST_DATA_PATH = 'give_credit/data/raw/cs-test.csv'
data1 = pd.read_csv(RAW_TRAINING_DATA_PATH)
data2 = pd.read_csv(RAW_TEST_DATA_PATH)

INTERIM_TEST_DATA = '/home/fakhri/pacmann_project/give_credit/data/interim/interim_test_forth_exp_tuned.csv'
INTERIM_TRAINING_DATA = '/home/fakhri/pacmann_project/give_credit/data/interim/interim_training_forth_exp_tuned.csv'
data3 = pd.read_csv(INTERIM_TRAINING_DATA)
data4 = pd.read_csv(INTERIM_TEST_DATA)
PARAMS_PATH = '/home/fakhri/pacmann_project/give_credit/src/experiment_config/fourth_exp_tuned.yaml'
def ramdomize_col_drop(data:pd.DataFrame) :
     cols = list(data.columns)
     random_idx = np.random.randint(low=0,high=len(cols)-1)
     col_name = cols[random_idx]
     print(col_name)
     return data.drop(col_name,axis=1)

    


@pytest.mark.parametrize(
    "data,training_req,expected_output",
    [
        (data1, True,True),
        (data2, False,True),
        (ramdomize_col_drop(data1), True,False),
        (ramdomize_col_drop(data2), True,False)
    ],
)
def test_wrangling_data(data,training_req,expected_output) : 
    assert isinstance(wrangling.wrangling_data(data,training_req=training_req,params_path=PARAMS_PATH),pd.DataFrame)  == expected_output
     

