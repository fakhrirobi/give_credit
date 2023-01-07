import yaml
from src.utils.path_checker import yaml_extension_checker


def load_yaml(params_path):
    """ """
    yaml_extension_checker(params_path)

    with open(params_path, "r") as stream:
        config = yaml.safe_load(stream)
    return config
