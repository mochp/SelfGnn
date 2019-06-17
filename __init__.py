import yaml

with open('./config.yml','rb') as file_config:
    config = yaml.load(file_config)
    print("load config")