import yaml

def load_config_yaml(file_path="config.yaml"):
    with open(file_path, 'r') as stream:
        config = yaml.safe_load(stream)
    return config