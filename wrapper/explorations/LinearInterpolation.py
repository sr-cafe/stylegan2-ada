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
		interpolator = easing(start.vector, end.vector, 1)

		for alpha in steps_num:
			vector = interpolator.ease(alpha)
			images.append(self.styleGanWrapper.from_z_vector(vector, truncation))

		return images
