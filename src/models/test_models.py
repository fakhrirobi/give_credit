import joblib 
import pytest
import pandas as pd 
import os
import numpy as np 
import config.path_config as path_config



MODEL_PATH = os.path.join(path_config.ROOT_DIR,'models','LGBMClassifier_fourth_exp_tuned.joblib')
SAMPLE_TEST_PATH = os.path.join(path_config.ROOT_DIR,'data','processed','processed_test_forth_exp_tuned.csv')




X_sample1 = pd.read_csv(SAMPLE_TEST_PATH)

@pytest.fixture(scope="module")
def model_load():
    model = joblib.load(MODEL_PATH)
    return model

def ramdomize_col_drop(data:pd.DataFrame) :
     cols = list(data.columns)
     random_idx = np.random.randint(low=0,high=len(cols)-1)
     col_name = cols[random_idx]
     print(col_name)
     return data.drop(col_name,axis=1)





@pytest.mark.parametrize(
    "data,features_in_model,array_out",
    [
        (X_sample1, True,True),
        (ramdomize_col_drop(X_sample1), False,False),

    ],
)
def test_model_output_types(data,features_in_model,array_out,model_load) :
    models_feature_names =  set(model_load.feature_name_)
    data_features_names = set(list(data.columns))
    compare = models_feature_names.symmetric_difference(data_features_names)==set()
    array_output = type(models._validate_data(data,validate_separately=False,reset=False))==np.array
    assert compare == features_in_model
    assert array_output ==array_out

@pytest.mark.parametrize(
    "data,features_in_model",
    [
        (X_sample1, True),
        (ramdomize_col_drop(X_sample1), False),

    ],
)
def test_model_output_types(data,features_in_model,model_load) :
    models_feature_names =  set(model_load.feature_name_)
    data_features_names = set(list(data.columns))
    compare = models_feature_names.symmetric_difference(data_features_names)==set()
    assert compare == features_in_model


