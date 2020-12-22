import numpy as np

class Latent:
	# TODO: Send network shape and check compatibility
	@staticmethod
	def from_file(filepath):
		vector = np.load(filepath)['dlatents']
		assert vector.shape[1:] == (18, 512) # [N, 18, 512]
		return Latent(vector)

	@staticmethod
	def to_file(latent, filepath):
		np.save(filepath, latent.vector)
		return latent

	def __init__(self, vector):
		self.vector = vector

