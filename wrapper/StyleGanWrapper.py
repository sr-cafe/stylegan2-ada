import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import pickle

from .GeneratedImage import GeneratedImage
from .Latent import Latent
from .NetworkLoader import NetworkLoader

class StyleGanWrapper:
	@staticmethod
	def expand_seed(seed, z_space_dims=512):
		rnd = np.random.RandomState(seed)
		return (rnd.randn(1, z_space_dims), rnd)

	def __init__(self, network_loader, truncation_psi=0.5):
		self.network_loader = network_loader
		self.truncation_psi = truncation_psi

	def load_network(self, network_path):
		self.network_path = network_path

		tflib.init_tf()

		self._G, self._D, self.Gs = self.network_loader.load(self.network_path)

		print(self.output_shape())

		self.noise_vars = [var for name, var in self.Gs.components.synthesis.vars.items() if name.startswith('noise')]
		self.Gs_kwargs = {
			'output_transform': dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True),
			'randomize_noise': False
		}

		return self

	def __generate(self, z, label=None):
		image = self.Gs.run(z, label, **self.Gs_kwargs) # [minibatch, height, width, channel]
		return GeneratedImage(image, Latent(z))

	def from_seed(self, seed, truncation_psi=None, class_idx=None):
		z, rnd = StyleGanWrapper.expand_seed(seed, *self.Gs.input_shape[1:]) # [minibatch, component]

		if truncation_psi is not None:
			truncation = truncation_psi
		else:
			truncation = self.truncation_psi

		if truncation is not None:
			self.Gs_kwargs['truncation_psi'] = truncation

		label = np.zeros([1] + self.Gs.input_shapes[1][1:])
		if class_idx is not None:
			label[:, class_idx] = 1

		tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in self.noise_vars}) # [height, width]
		return self.__generate(z, label)

	def from_latent(self, latent):
		if isinstance(latent, str):
			latent = Latent.from_file(latent, self.output_shape())
		elif not isinstance(latent, Latent):
			latent = Latent(latent)
		imgs = self.Gs.components.synthesis.run(latent.vector, output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True))
		return imgs

	def output_shape(self):
		return self.Gs.output_shape
