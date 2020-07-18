import json


class OBJ_DICT(dict):
	def __init__(self, *args, **kwargs):
		super(OBJ_DICT, self).__init__(*args, **kwargs)
		for arg in args:
			if isinstance(arg, dict):
				for k, v in arg.items():
					self[k] = v
		if kwargs:
			for k, v in kwargs.items():
				self[k] = v

	def __getattr__(self, attr):
		return self.get(attr)

	def __setattr__(self, key, value):
		self.__setitem__(key, value)

	def __setitem__(self, key, value):
		super(OBJ_DICT, self).__setitem__(key, value)
		self.__dict__.update({key: value})

	def __delattr__(self, item):
		self.__delitem__(item)

	def __delitem__(self, key):
		super(OBJ_DICT, self).__delitem__(key)
		del self.__dict__[key]


class CONFIG(OBJ_DICT):
	def __init__(self, config_file, **kwargs):

		with open(config_file) as f:
			config = json.load(f)

		for key, val in kwargs.items():
			config[key] = val

		super(CONFIG, self).__init__(config)