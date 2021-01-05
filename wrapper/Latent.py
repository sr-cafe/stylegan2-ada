import pickle
import numpy as np

class Latent:
	# TODO: Check file and network shapes compatibility
	@staticmethod
	def load(filepath, network_output_shape):
		with open(filepath, 'rb') as pf:
			latent = pickle.load(pf)
		return latent

	@staticmethod
	def from_seed(seed, truncation_psi=0.5, z_space_dims=512):
		rnd = np.random.RandomState(seed)
		return Latent(rnd.randn(1, z_space_dims), truncation_psi, seed, 'z')

	def __init__(self, vector, truncation_psi, seed = 1, type = 'z'):
		self.vector = vector
		self.truncation_psi = truncation_psi
		self.seed = seed
		self.type = type

	def save(self, filepath):
		with open(filepath, 'wb') as f:
			pickle.dump(self, f)
		return self
