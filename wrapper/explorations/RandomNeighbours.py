import numpy as np
from ..Latent import Latent
from .BaseExploration import BaseExploration

class RandomNeighbours(BaseExploration):

	def __init__(self, styleGanWrapper):
		BaseExploration.__init__(self, styleGanWrapper)

	def explore(self, latent, radius, num_samples):
		images = [self.styleGanWrapper.from_latent(latent)]

		for indx in range(num_samples):
			random = np.random.uniform(-radius,radius,[1,512])
			vector = np.clip(np.add(latent.vector, random), -1, 1)
			currentLatent = Latent(vector, latent.truncation_psi)
			images.append(self.from_latent(currentLatent))

		return images

