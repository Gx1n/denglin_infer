import os


def extract_model_suffix(model_path):
    return os.path.splitext(model_path)[-1]


def extract_model_name(model_path):
    return os.path.basename(model_path)