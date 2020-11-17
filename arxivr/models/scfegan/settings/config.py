import json

class DICTIONARY:
    def __init__(self, **response):
        for k,v in response.items():
            if isinstance(v, dict):
                self.__dict__[k] = DICTIONARY(**v)
            else:
                self.__dict__[k] = v


class CONFIG(DICTIONARY):
    def __init__(self, config_file, **kwargs):

        with open(config_file) as f:
            config = json.load(f)

        self.config = config

        for key, val in kwargs.items():
            config[key] = val

        super(CONFIG, self).__init__(**config)