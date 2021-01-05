import numpy as np
from ..Latent import Latent
from .BaseExploration import BaseExploration
from .Easings import *

class TruncationInterpolation(BaseExploration):
	def __init__(self, styleGanWrapper):
		BaseExploration.__init__(self, styleGanWrapper)

	def run(self, seed, start, end, steps, easing=LinearInOut):
		step_inc = 1.0 / steps
		steps_num = np.arange(0, 1, step_inc)
		images = []

		interpolator = easing(start, end, 1)

		for alpha in steps_num:
			truncation = interpolator.ease(alpha)
			images.append(self.styleGanWrapper.from_latent(Latent.from_seed(seed, truncation)))

		return images
