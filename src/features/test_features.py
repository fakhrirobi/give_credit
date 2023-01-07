import pandas as pd 
import numpy as np
import pytest
import os 
import src.features.feature_eng as feature_eng
import config.path_config as path_config

INTERIM_TEST_DATA = os.path.join(path_config.ROOT_DIR,'data','interim','interim_test_forth_exp_tuned.csv')
INTERIM_TRAINING_DATA = os.path.join(path_config.ROOT_DIR,'data','interim','interim_training_forth_exp_tuned.csv')
data3 = pd.read_csv(INTERIM_TRAINING_DATA)
data4 = pd.read_csv(INTERIM_TEST_DATA)
PARAMS_PATH = os.path.join(path_config.ROOT_DIR,'src','experiment_config','fourth_exp_tuned.yaml')
def ramdomize_col_drop(data:pd.DataFrame) :
     cols = list(data.columns)
     random_idx = np.random.randint(low=0,high=len(cols)-1)
     col_name = cols[random_idx]
     print(col_name)
     return data.drop(col_name,axis=1)







@pytest.mark.parametrize(
    "data,expected_output",
    [
        (data3,True),
        (data4,True),
        (ramdomize_col_drop(data3),False),
        (ramdomize_col_drop(data4),False)

    ],
)
def test_feature_engineering_data(data,expected_output) : 
    assert isinstance(feature_eng.feature_engineering_process(data,params_path=PARAMS_PATH),pd.DataFrame)  == expected_output

