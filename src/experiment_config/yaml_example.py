import yaml

with open(
    "/home/fakhri/give_me_credit/give_me_credit/give_me_credit/src/params_dir/first_experiment.yaml",
    "r",
) as stream:
    data_loaded = yaml.safe_load(stream)

data_loaded["params"]
