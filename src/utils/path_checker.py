import os


def csv_extension_checker(path: os.PathLike):
    file_extension = os.path.splitext(path)[1]

    if file_extension != ".csv":
        raise ValueError(f"File format should be .csv. Provided {path}")
    

def yaml_extension_checker(path: os.PathLike):
    file_extension = os.path.splitext(path)[1]
    required_ext = [".yaml", ".yml"]
    if file_extension not in required_ext:
        raise ValueError(f"File format should be {required_ext}. Provided {path}")
