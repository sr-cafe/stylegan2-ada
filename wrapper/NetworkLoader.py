import os
import gdown
import pickle

class NetworkLoader:
	def __init__(self, output_dir):
		self.output_dir = output_dir
		os.makedirs(self.output_dir, exist_ok=True)

	# TODO: Make this compatible with other sources than GDrive.
	# TODO: Make saved_network_name optional
	# Currently "network_path" is expected to be a GDrive file id.
	def load(self, network_path, saved_network_name, output_dir=None):
		url = f'https://drive.google.com/uc?id={network_path}'

		if output_dir is not None:
			dir = output_dir
		else:
			dir = self.output_dir

		output = f'{dir}/{saved_network_name}'

		pickle_file = gdown.download(url, output, False)

		with open(pickle_file) as pf:
			_G, _D, Gs = pickle.load(pf)

		return _G, _D, Gs
