import json


class Config:
    def __init__(self, config_dict):
        self.__dict__.update(config_dict)

    def write_to_file(self, file_path):
        with open(file_path, 'w') as file:
            # Convert the configuration dictionary to a JSON-formatted string
            # and write it to the file
            json.dump(self.__dict__, file, indent=4)


def load_config(config_file):
    with open(config_file, 'r') as file:
        config_dict = json.load(file)
    return Config(config_dict)
