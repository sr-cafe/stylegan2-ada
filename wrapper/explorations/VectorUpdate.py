import numpy as np
from ..Latent import Latent
from .BaseExploration import BaseExploration

class VectorUpdate(BaseExploration):
	def __init__(self, styleGanWrapper):
		BaseExploration.__init__(self, styleGanWrapper)

	def run(self, latent, transformFunction, cumulative=True):
		images = [self.styleGanWrapper.from_latent(latent)]
		currentLatent = latent
		dimensions = 512
		for i in np.arange(0, dimensions, 1):
			if not cumulative:
				currentLatent = Latent(latent.vector.copy(), latent.truncation_psi, latent.type)
			currentLatent.vector[0][i] = transformFunction(currentLatent.vector[0][i], i, dimensions)
			images.append(self.styleGanWrapper.from_latent(currentLatent))

		return images

