import numpy as np
from ..Latent import Latent
from .BaseExploration import BaseExploration

class LinearInterpolation(BaseExploration):
	def __init__(self, styleGanWrapper):
		BaseExploration.__init__(self, styleGanWrapper)

	def run(self, latents, steps, shuffle=False):
		if shuffle:
			np.random.shuffle(latents)

		images = []
		truncation = latents[0].truncation_psi
		latentType = latents[0].type

		for i in range(len(latents) - 1):
			for j in range(steps):
				fraction = j / float(steps)
				vector = latents(i + 1) * fraction + latents[i] * (1 - fraction)
				currentLatent = Latent(vector, truncation, latentType)
				images.append(self.styleGanWrapper.from_latent(currentLatent))
