import json


class Config:
    def __init__(self, config_dict):
        self.__dict__.update(config_dict)


def load_config(config_file):
    with open(config_file, 'r') as file:
        config_dict = json.load(file)
    return Config(config_dict)
