import pickle
import numpy as np

class Latent:
	# TODO: Throw error when file and network shapes aren't equal
	@staticmethod
	def from_file(filepath, network_output_shape):
		vector = np.load(filepath, allow_pickle=True)['dlatents']
		max_l = 2 * int(np.log2(network_output_shape[-1]) - 1)  # max_l=18 for 1024x1024 models
		if vector.shape[1:] != (max_l, 512):  # [N, max_l, 512]
			print('Error. Shapes are not compatible')
		return Latent(vector)

	def __init__(self, vector, truncation_psi, type = 'z'):
		self.vector = vector
		self.truncation_psi = truncation_psi
		self.type = type

	def save(self, filepath):
		with open(filepath, 'wb') as f:
			pickle.dump(self, f)
		return self
