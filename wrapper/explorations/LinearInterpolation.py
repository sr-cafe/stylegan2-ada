import numpy as np
from ..Latent import Latent
from .BaseExploration import BaseExploration
from .Easings import *

class LinearInterpolation(BaseExploration):
	def __init__(self, styleGanWrapper):
		BaseExploration.__init__(self, styleGanWrapper)

	def run(self, start, end, steps, easing=LinearInOut):
		step_inc = 1.0 / steps
		steps_num = np.arange(0, 1, step_inc)
		images = []
		truncation = start.truncation_psi
		interpolator = easing(start, end, 1)

		for alpha in steps_num:
			currentLatent = Latent(interpolator.ease(alpha), truncation)
			images.append(self.from_latent(currentLatent))

		return images
