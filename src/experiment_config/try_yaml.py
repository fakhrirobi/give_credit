from src.utils.load_config import load_yaml


path = "give_me_credit/give_me_credit/src/experiment_config/fourth_exp_tuned.yaml"

config = load_yaml(params_path=path)

config.get("data_wrangling_output_dtypes")
