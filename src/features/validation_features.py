import os
import pandas as pd
import src.utils.load_config as load_config
import src.utils.path_checker as path_checker


TARGET_COL = "SeriousDlqin2yrs"


def validate_wrangling_output_col(
    data: pd.DataFrame,
    params_path: os.PathLike,
):
    """_summary_

    Args:
        feature_names (list): features that provided in inference input data
        config (list): features according to experiment .yaml file
    """
    path_checker.yaml_extension_checker(params_path)
    config = load_config.load_yaml(params_path)

    data_wrangling_output_dtypes = config.get("data_wrangling_output_dtypes")
    # print('dict:',data_wrangling_output_dtypes)
    # col_dtypes_reference = dict(ChainMap(*data_wrangling_output_dtypes))
    reference_col_names = list(data_wrangling_output_dtypes.keys())
    differences = list(
        set(reference_col_names).symmetric_difference(set(data.columns.tolist()))
    )
    print("differences : ", differences)
    for dif in differences:
        if dif == TARGET_COL:
            continue
        elif dif != TARGET_COL:
            raise ValueError(
                f"Found unexpected column : {dif} in dataset, check data wrangling steps in data/wrangling.py"
            )

    # check dtypes :
    for col in data.columns:
        reference_dtypes = data_wrangling_output_dtypes.get(col)
        if data[col].dtypes not in reference_dtypes:
            raise ValueError(
                f"Found not same dtypes at column{col}.It supposed to be dtpes {reference_dtypes}.Provided {data[col].dtypes}"
            )


def validate_feature_engineering_output_col(
    data: pd.DataFrame, params_path: os.PathLike
):

    path_checker.yaml_extension_checker(params_path)
    config = load_config.load_yaml(params_path)
    feature_engineering_output_dtypes = config.get("feature_eng_model_input")
    reference_col_names = list(feature_engineering_output_dtypes.keys())

    differences = list(
        set(reference_col_names).symmetric_difference(data.columns.tolist())
    )
    for dif in differences:
        if dif == TARGET_COL:
            continue
        elif dif != TARGET_COL:
            raise ValueError(
                f"Found unexpected column : {dif} in dataset, check data wrangling steps in data/wrangling.py"
            )

    # check dtypes :
    for col in data.columns:
        print('validation',col)
        reference_dtypes = feature_engineering_output_dtypes.get(col)
        if data[col].dtypes not in reference_dtypes:
            raise ValueError(
                f"Found not same dtypes at column{col}.It supposed to be dtpes {reference_dtypes}.Provided {data[col].dtypes}"
            )
